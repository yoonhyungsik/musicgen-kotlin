package com.example.musicgen

import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.random.Random
import org.json.JSONArray
import org.json.JSONObject
import java.nio.LongBuffer

class MusicGenGenerator(
    private val decoderModelPath: String,
    private val vocoderModelPath: String,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
) {
    private val decoderSession: OrtSession = env.createSession(decoderModelPath, OrtSession.SessionOptions())
    private val vocoderSession: OrtSession = env.createSession(vocoderModelPath, OrtSession.SessionOptions())

    // === STEP 1: Tokenizer ===
    fun simpleTokenizer(prompt: String, vocab: Map<String, Int>): LongArray {
        val tokens = mutableListOf<Long>()
        val words = prompt.lowercase().replace(Regex("[^a-z0-9']"), " ").split(" ")
        for (word in words) {
            if (word.isNotBlank()) {
                val token = vocab[word]
                if (token != null) {
                    tokens.add(token.toLong())
                } else {
                    tokens.add(vocab["<unk>"]?.toLong() ?: 0L)
                }
            }
        }
        return tokens.toLongArray()
    }

    fun loadVocabFromTokenizerJson(file: File): Map<String, Int> {
        val raw = file.readText(StandardCharsets.UTF_8)
        val json = JSONObject(raw)
        val vocab = mutableMapOf<String, Int>()

        // 1. Handle vocab as JSONObject (normal case)
        if (json.has("model")) {
            val model = json.getJSONObject("model")
            if (model.has("vocab") && model.get("vocab") is JSONObject) {
                val modelVocab = model.getJSONObject("vocab")
                for (key in modelVocab.keys()) {
                    vocab[key] = modelVocab.getInt(key)
                }
            }
        }

        // 2. Handle added_tokens array (for special tokens)
        if (json.has("added_tokens")) {
            val addedTokens = json.getJSONArray("added_tokens")
            for (i in 0 until addedTokens.length()) {
                val obj = addedTokens.getJSONObject(i)
                vocab[obj.getString("content")] = obj.getInt("id")
            }
        }

        return vocab
    }

    // === STEP 2: Transformer Decoder 실행 ===
    fun generateTokens(
        inputIds: LongArray,
        encoderHiddenStates: Array<Array<FloatArray>>,
        encoderAttentionMask: Array<LongArray>,
        maxNewTokens: Int = 32,
        guidanceScale: Float = 1.0f,
        negativeEncoderHiddenStates: Array<Array<FloatArray>>? = null,
        numCodebooks: Int = 4
    ): Array<LongArray> {
        val sequence = mutableListOf<Long>()
        sequence.addAll(inputIds.toList())

        repeat(maxNewTokens) {
            val flatInputIds = LongArray(numCodebooks * sequence.size) { sequence[it % sequence.size] }
            val flatBuffer = LongBuffer.wrap(flatInputIds)
            val inputTensor = OnnxTensor.createTensor(env, flatBuffer, longArrayOf((numCodebooks).toLong(), sequence.size.toLong()))

            val encoderTensor = OnnxTensor.createTensor(env, encoderHiddenStates)
            val attentionTensor = OnnxTensor.createTensor(env, encoderAttentionMask)

            val inputs = mapOf(
                "input_ids" to inputTensor,
                "encoder_hidden_states" to encoderTensor,
                "encoder_attention_mask" to attentionTensor
            )

            val output = decoderSession.run(inputs)
            val logits = (output[0].value as Array<Array<FloatArray>>)[0] // [1, T, V]
            val lastLogits = logits[logits.size - 1]

            val adjustedLogits = if (guidanceScale != 1.0f && negativeEncoderHiddenStates != null) {
                val negEncoderTensor = OnnxTensor.createTensor(env, negativeEncoderHiddenStates)
                val negInputs = mapOf(
                    "input_ids" to inputTensor,
                    "encoder_hidden_states" to negEncoderTensor,
                    "encoder_attention_mask" to attentionTensor
                )
                val negOutput = decoderSession.run(negInputs)
                val negLogits = (negOutput[0].value as Array<Array<FloatArray>>)[0].last()
                lastLogits.zip(negLogits).map { (pos, neg) -> pos + guidanceScale * (pos - neg) }.toFloatArray()
            } else {
                lastLogits
            }

            val nextToken = sampleFromLogits(adjustedLogits)
            sequence.add(nextToken)
        }
        return arrayOf(sequence.toLongArray())
    }

    // === STEP 3: Top-k Sampling ===
    private fun sampleFromLogits(logits: FloatArray, topK: Int = 8): Long {
        val indexed = logits.mapIndexed { idx, value -> idx to value }
        val top = indexed.sortedByDescending { it.second }.take(topK)

        val maxLogit = top.maxOf { it.second }
        val probs = top.map { exp(it.second - maxLogit) }
        val sum = probs.sum()
        val normProbs = probs.map { it / sum }

        val r = Random.nextFloat()
        var acc = 0f
        for ((i, p) in normProbs.withIndex()) {
            acc += p
            if (r < acc) return top[i].first.toLong()
        }
        return top[0].first.toLong() // fallback
    }

    // === STEP 4: Reshape & Run Vocoder ===
    fun runVocoder(tokenIds: Array<LongArray>): FloatArray {
        val chunkLength = tokenIds[0].size
        val codebooks = Array(1) { Array(1) { Array(4) { LongArray(chunkLength) } } }
        for (i in 0 until chunkLength) {
            for (cb in 0 until 4) {
                codebooks[0][0][cb][i] = tokenIds[0][i] // 간단한 복제 방식
            }
        }

        val inputTensor = OnnxTensor.createTensor(env, codebooks)
        val output = vocoderSession.run(mapOf("audio_codes" to inputTensor))
        val audio = (output[0].value as Array<Array<FloatArray>>)[0][0] // [B, 1, audio_len]
        return audio
    }

    // === STEP 5: WAV 저장 ===
    fun saveWav(file: File, waveform: FloatArray, sampleRate: Int = 32000) {
        val byteRate = sampleRate * 4
        val dataSize = waveform.size * 4
        val totalSize = 36 + dataSize

        val header = ByteBuffer.allocate(44)
        header.order(ByteOrder.LITTLE_ENDIAN)
        header.put("RIFF".toByteArray())
        header.putInt(totalSize)
        header.put("WAVE".toByteArray())
        header.put("fmt ".toByteArray())
        header.putInt(16) // Subchunk1Size (PCM)
        header.putShort(3) // AudioFormat 3 = IEEE float
        header.putShort(1) // NumChannels
        header.putInt(sampleRate)
        header.putInt(byteRate)
        header.putShort(4) // BlockAlign
        header.putShort(32) // BitsPerSample
        header.put("data".toByteArray())
        header.putInt(dataSize)

        val out = FileOutputStream(file)
        out.write(header.array())

        val audioBytes = ByteBuffer.allocate(dataSize)
        audioBytes.order(ByteOrder.LITTLE_ENDIAN)
        waveform.forEach { audioBytes.putFloat(it) }

        out.write(audioBytes.array())
        out.flush()
        out.close()
    }
}
