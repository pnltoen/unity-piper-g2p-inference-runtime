using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;
using Unity.InferenceEngine;
using System.Collections;
using FF = Unity.InferenceEngine.Functional;
using UnityEngine.InputSystem;
using TMPro;
using UnityEngine.UI;

public class Piper : MonoBehaviour
{
    Dictionary<string, int> vocab = new Dictionary<string, int>();
    Dictionary<int, string> idToToken = new Dictionary<int, string>();
    Dictionary<string, string> arpabetToIPA = new Dictionary<string, string>();
    Dictionary<string, int> IPAToIDs = new Dictionary<string, int>();

    public TMP_InputField inputField;

    Model bartEncoder;
    Worker bartEncoder_Worker;

    Model bartDecoder;
    Worker bartDecoder_Worker;

    Model piper;
    Worker piper_Worker;

    Worker argmax;

    Tensor<int> inputIDs_encoder;
    Tensor<int> attentionMask_bart;
    Tensor<float> encoder_hidden_states;

    int tokenCount = 0;

    const int START_TOKEN = 0;
    const int END_TOKEN = 2;
    const int MAX_DECODE = 128;

    const BackendType backend = BackendType.CPU;

    List<string> tokens;
    List<List<string>> punctuationSlots;
    int[] decoderTokens;

    List<int> allPhonemeTokenIDs = new List<int>();

    // Set a larger number for faster GPUs
    public int k_LayersPerFrame = 250;

    IEnumerator piper_Async;

    string defaultVoice = "en_US_lessac_high.sentis";
    public float[] scales = { 0.4f, 1.1f, 0.6f }; //[noise_scale, length_scale, noise_w]

    void Start()
    {
        LoadPretrainedData();
        LoadBartModel();
        LoadPiperModel(defaultVoice);
        argMax();
    }

    public void SwitchSpeaker(Toggle toggle)
    {
        if (toggle.isOn)
        {
            string voice = toggle.gameObject.name + ".sentis";
            LoadPiperModel(voice);
        }
    }

    public void GenerateTTS(string _)
    {
        string input = inputField.text;
        Tokenizer(input);
        StartCoroutine(RunModel(tokens));
    }

    void LoadBartModel()
    {
        bartEncoder = ModelLoader.Load(Application.streamingAssetsPath + "/encoder_model.sentis");
        bartEncoder_Worker = new Worker(bartEncoder, backend);

        bartDecoder = ModelLoader.Load(Application.streamingAssetsPath + "/decoder_model.sentis");
        bartDecoder_Worker = new Worker(bartDecoder, backend);
    }

    void LoadPiperModel(string voice)
    {
        string piper_Path = Path.Combine(Application.streamingAssetsPath, voice);
        piper_Worker?.Dispose();
        piper = ModelLoader.Load(piper_Path);
        piper_Worker = new Worker(piper, BackendType.GPUCompute);
    }

    void LoadPretrainedData()
    {
        string Vocabpath = Application.streamingAssetsPath + "/g2p_vocab.json";
        string IPApath = Application.streamingAssetsPath + "/piper_IPA.json";
        string IPAIDspath = Application.streamingAssetsPath + "/piper_IDs.json";

        // Vocab Load
        if (!File.Exists(Vocabpath))
        {
            return;
        }

        try
        {
            string jsonText = File.ReadAllText(Vocabpath);
            vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(jsonText);
            idToToken = vocab.ToDictionary(kv => kv.Value, kv => kv.Key);
        }
        catch (Exception e)
        {
        }

        // IPA Mapping Load
        if (!File.Exists(IPApath))
        {
            return;
        }

        try
        {
            string ipaJson = File.ReadAllText(IPApath);
            arpabetToIPA = JsonConvert.DeserializeObject<Dictionary<string, string>>(ipaJson);
        }
        catch (Exception e)
        {
        }

        if (!File.Exists(IPAIDspath))
        {
            return;
        }

        try
        {
            string ipaIdJson = File.ReadAllText(IPAIDspath);
            IPAToIDs = JsonConvert.DeserializeObject<Dictionary<string, int>>(ipaIdJson);
        }
        catch (Exception e)
        {
        }

    }

    void argMax()
    {
        FunctionalGraph graph = new FunctionalGraph();
        var input = graph.AddInput(DataType.Float, new DynamicTensorShape(-1, -1, -1));
        var amax = FF.ArgMax(input, -1, false);
        var selectTokenModel = graph.Compile(amax);
        argmax = new Worker(selectTokenModel, backend);
    }

    public void Tokenizer(string sentence)
    {
        tokens = new List<string>();
        punctuationSlots = new List<List<string>>();

        string[] raw = sentence.ToLower().TrimEnd().Split(' ');

        for (int i = 0; i < raw.Length; i++)
        {
            string word = raw[i];
            List<string> punctuations = new List<string>();

            if (!string.IsNullOrWhiteSpace(word))
            {
                for (int j = word.Length - 1; j >= 0; j--)
                {
                    if (!char.IsLetterOrDigit(word[j]))
                    {
                        punctuations.Add(word[j].ToString());
                    }
                    else
                    {
                        word = word.Substring(0, j + 1);
                        break;
                    }
                }
            }

            tokens.Add(word);
            punctuationSlots.Add(punctuations);
        }
    }

    (Tensor<float>, Tensor<int>) RunEncoder(string word)
    {
        string[] chars = word.Select(c => c.ToString()).ToArray();
        int[] middle = chars.Select(t => vocab.ContainsKey(t) ? vocab[t] : vocab["<unk>"]).ToArray();

        int[] inputIDs = new int[middle.Length + 2];
        inputIDs[0] = vocab["<s>"];
        Array.Copy(middle, 0, inputIDs, 1, middle.Length);
        inputIDs[inputIDs.Length - 1] = vocab["</s>"];

        int[] attentionMask = Enumerable.Repeat(1, inputIDs.Length).ToArray();

        var inputIDs_encoder = new Tensor<int>(new TensorShape(1, inputIDs.Length), inputIDs);
        attentionMask_bart = new Tensor<int>(new TensorShape(1, attentionMask.Length), attentionMask);

        bartEncoder_Worker.Schedule(inputIDs_encoder, attentionMask_bart);
        encoder_hidden_states = bartEncoder_Worker.PeekOutput("last_hidden_state").ReadbackAndClone() as Tensor<float>;
        inputIDs_encoder.Dispose();

        return (encoder_hidden_states, attentionMask_bart);
    }

    (int[], int) RunDecoder(Tensor<float> encoder_hidden_states, Tensor<int> attentionMask_bart)
    {
        decoderTokens = new int[MAX_DECODE];
        decoderTokens[0] = END_TOKEN;
        decoderTokens[1] = START_TOKEN;
        tokenCount = 2;

        for (int step = 0; step < MAX_DECODE; step++)
        {
            var decoder_input_slice = decoderTokens.Take(tokenCount).ToArray();
            var decoder_input_tensor = new Tensor<int>(new TensorShape(1, tokenCount), decoder_input_slice);

            bartDecoder_Worker.Schedule(attentionMask_bart, decoder_input_tensor, encoder_hidden_states);
            var logits = bartDecoder_Worker.PeekOutput("logits") as Tensor<float>;
            using var cpuLogits = logits.ReadbackAndClone() as Tensor<float>;
            logits.Dispose();

            int vocabSize = cpuLogits.shape[2];
            float[] lastLogits = new float[vocabSize];
            for (int i = 0; i < vocabSize; i++)
                lastLogits[i] = cpuLogits[0, tokenCount - 1, i];

            var lastLogitsTensor = new Tensor<float>(new TensorShape(1, 1, vocabSize), lastLogits);

            argmax.Schedule(lastLogitsTensor);

            using var t_Token = argmax.PeekOutput().ReadbackAndClone() as Tensor<int>;
            int nextToken = t_Token[0];

            decoderTokens[tokenCount] = nextToken;
            tokenCount++;

            lastLogitsTensor.Dispose();
            decoder_input_tensor.Dispose();

            if (nextToken == END_TOKEN)
                break;
        }
        encoder_hidden_states.Dispose();
        attentionMask_bart.Dispose();

        return (decoderTokens, tokenCount);
    }

    public List<int> DecodeToIPAIDs(int[] decoderTokens, int tokenCount, int token_order)
    {
        int[] validTokenIDs = decoderTokens
            .Take(tokenCount)
            .Where(id => id != 0 && id != 2)
            .ToArray();

        List<string> arpabetTokens = new List<string>();
        foreach (int id in validTokenIDs)
        {
            if (idToToken.TryGetValue(id, out string token))
                arpabetTokens.Add(token);
        }

        List<string> ipaTokens = new List<string>();
        foreach (string arp in arpabetTokens)
        {
            if (arpabetToIPA.TryGetValue(arp, out string ipa))
                ipaTokens.Add(ipa);
        }

        ipaTokens.AddRange(punctuationSlots[token_order]);

        foreach (string ipa in ipaTokens)
        {
            foreach (char c in ipa)
            {
                if (IPAToIDs.TryGetValue(c.ToString(), out int id))
                {
                    allPhonemeTokenIDs.Add(id);
                }
            }
            allPhonemeTokenIDs.Add(0);
        }

        allPhonemeTokenIDs.Add(3);
        allPhonemeTokenIDs.Add(0);

        return allPhonemeTokenIDs;
    }

    IEnumerator RunPiper()
    {
        int count = allPhonemeTokenIDs.Count;
        allPhonemeTokenIDs.Insert(0, 1);  // START
        allPhonemeTokenIDs.RemoveAt(allPhonemeTokenIDs.Count - 1);
        allPhonemeTokenIDs[allPhonemeTokenIDs.Count - 2] = 2;
        var phonemes_Ids = allPhonemeTokenIDs.ToArray();
        allPhonemeTokenIDs.Clear();

        var piper_input = new Tensor<int>(new TensorShape(1, phonemes_Ids.Length), phonemes_Ids);
        var piper_input_lengths = new Tensor<int>(new TensorShape(1), new int[] { phonemes_Ids.Length });
        var piper_scales = new Tensor<float>(new TensorShape(scales.Length), scales);

        piper_Async = piper_Worker.ScheduleIterable(piper_input, piper_input_lengths, piper_scales);

        int it = 0;
        while (piper_Async.MoveNext())
        {
            if (++it % k_LayersPerFrame == 0)
                yield return null;
        }
        yield return null;

        var piper_output = piper_Worker.PeekOutput("output") as Tensor<float>; // batch x Channel x Height x Time
        var awaiter = piper_output.ReadbackAndCloneAsync().GetAwaiter();

        awaiter.OnCompleted(() =>
        {
            var piper_CPUoutput = awaiter.GetResult().DownloadToArray();
            PlayAndSavePiperAudio(piper_CPUoutput);

            piper_output.Dispose();
            piper_input.Dispose();
            piper_input_lengths.Dispose();
            piper_scales.Dispose();
        });
    }

    IEnumerator RunModel(List<string> tokens)
    {
        for (int i = 0; i < tokens.Count; i++)
        {
            RunEncoder(tokens[i]);
            RunDecoder(encoder_hidden_states, attentionMask_bart);
            DecodeToIPAIDs(decoderTokens, tokenCount, i);

            yield return null;
        }
        StartCoroutine(RunPiper());
    }

    void PlayAndSavePiperAudio(float[] output)
    {
        AudioClip clip = AudioClip.Create("PiperAudio", output.Length, 1, 22050, false);

        clip.SetData(output, 0);

        AudioSource audioSource = GetComponent<AudioSource>();
        if (audioSource == null)
            audioSource = gameObject.AddComponent<AudioSource>();

        audioSource.clip = clip;
        audioSource.Play();
    }


    void OnDestroy()
    {
        argmax?.Dispose();
        inputIDs_encoder?.Dispose();
        attentionMask_bart?.Dispose();
        encoder_hidden_states?.Dispose();
        bartEncoder_Worker?.Dispose();
        bartDecoder_Worker?.Dispose();
        piper_Worker?.Dispose();
    }
}
