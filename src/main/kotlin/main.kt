import com.example.musicgen.MusicGenGenerator
import java.io.File

fun main() {
    val generator = MusicGenGenerator(
        decoderModelPath = "src/main/resources/decoder_model.onnx",
        vocoderModelPath = "src/main/resources/encodec_decode.onnx"
    )

    val vocab = generator.loadVocabFromTokenizerJson(File("src/main/resources/tokenizer.json"))
    val inputIds = generator.simpleTokenizer("a peaceful ambient melody", vocab)

    val encoderStates = Array(1) { Array(20) { FloatArray(768) { 0.01f } } }
    val attentionMask = Array(1) { LongArray(20) { 1L } }

    val tokens = generator.generateTokens(inputIds, encoderStates, attentionMask)
    val audio = generator.runVocoder(tokens)

    generator.saveWav(File("generated_music.wav"), audio)
    println("✅ Music generated and saved to 'generated_music.wav'")
}