package musicgen

import com.google.gson.Gson
import musicgen.tokenizer.TokenizerModel
import musicgen.tokenizer.UnigramTokenizer
import musicgen.inference.MusicGenGenerator
import java.io.FileNotFoundException

fun loadTokenizer(): TokenizerModel {
    val inputStream = object {}.javaClass.getResourceAsStream("/tokenizer.json")
        ?: throw FileNotFoundException("Could not find tokenizer.json in resources")
    val jsonStr = inputStream.bufferedReader().readText()
    return Gson().fromJson(jsonStr, TokenizerModel::class.java)
}

fun preprocessText(input: String): String {
    return input
        .lowercase()
        .replace(" ", "▁")
}

fun padOrTruncateToLength(tokenIds: List<Int>, targetLength: Int, padTokenId: Int): LongArray {
    return when {
        tokenIds.size == targetLength -> tokenIds.map { it.toLong() }.toLongArray()
        tokenIds.size < targetLength -> (tokenIds + List(targetLength - tokenIds.size) { padTokenId }).map { it.toLong() }.toLongArray()
        else -> tokenIds.take(targetLength).map { it.toLong() }.toLongArray()
    }
}

fun findPadTokenId(tokenizer: TokenizerModel): Int {
    return tokenizer.added_tokens.firstOrNull { it.content == "<pad>" }?.id
        ?: error("<pad> token not found in tokenizer")
}

fun main() {
    val tokenizer = loadTokenizer()
    val vocabMap = UnigramTokenizer.buildVocabMap(tokenizer.model.vocab)
    val inputText = preprocessText("a peaceful ambient melody with soft synth, gentle rhythm, and relaxing atmosphere")
    val tokenIdsRaw = UnigramTokenizer.tokenize(inputText, vocabMap)

    println("Token count (before padding): ${tokenIdsRaw.size}")

    val padId = findPadTokenId(tokenizer)
    val tokenIds = padOrTruncateToLength(tokenIdsRaw, targetLength = 1088, padTokenId = padId)

    val generator = MusicGenGenerator(
        decoderModelPath = "src/main/resources/decoder_model.onnx",
        vocoderModelPath = "src/main/resources/encodec_decode.onnx"
    )
    generator.generateMusic(tokenIds)
}
