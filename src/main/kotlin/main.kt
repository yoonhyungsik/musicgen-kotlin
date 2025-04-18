package musicgen

import ai.onnxruntime.*
import musicgen.inference.*
import musicgen.tokenizer.*

fun main() {
    val env = OrtEnvironment.getEnvironment()

    // Load ONNX sessions
    val decoderSession = env.createSession("src/main/resources/decoder_model.onnx")
    val vocoderSession = env.createSession("src/main/resources/encodec_decode.onnx")

    
    val decoder = DecoderRunner(decoderSession)
    val vocoder = VocoderRunner(vocoderSession)

    // Generate random input_ids: [4 x 192] = 768 tokens
    val numCodebooks = 4
    val sequenceLength = 192
    val shape = longArrayOf(numCodebooks.toLong(), sequenceLength.toLong())
    val inputIds = LongArray(numCodebooks * sequenceLength) { (0..255).random().toLong() }

    println("Generating audio from random latent tokens...")
    val tokenIds = decoder.decode(inputIds, shape)
    val audio = vocoder.synthesize(tokenIds)
    vocoder.saveAsWav(audio)

    println("✅ Music generated and saved to 'generated_music.wav'")
}