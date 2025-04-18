package musicgen.tokenizer

import com.google.gson.Gson
import java.io.File

object TokenizerJsonLoader {

    data class TokenizerJsonModel(
        val model: ModelSection,
        val added_tokens: List<SpecialToken>
    ) {
        data class ModelSection(
            val vocab: List<List<Any>>,  // token-logprob pair
            val merges: List<String>?
        )

        data class SpecialToken(
            val id: Int,
            val content: String
        )
    }

    fun loadFromTokenizerJson(path: String): Triple<Map<String, Int>, List<Pair<String, String>>, Map<String, Int>> {
        val jsonStr = File(path).readText(Charsets.UTF_8)
        val parsed = Gson().fromJson(jsonStr, TokenizerJsonModel::class.java)

        val vocabMap = mutableMapOf<String, Int>()

        parsed.model.vocab.forEachIndexed { index, entry ->
            val token = entry[0] as String
            vocabMap[token] = index
        }

        parsed.added_tokens.forEach { special ->
            vocabMap[special.content] = special.id
        }

        val merges = parsed.model.merges?.map {
            val parts = it.split(" ")
            parts[0] to parts[1]
        } ?: emptyList()

        val specialTokenMap = parsed.added_tokens.associate { it.content to it.id }

        vocabMap.keys.filter { it.contains("<") || it.contains(">") }.forEach { println(it) }

        return Triple(vocabMap, merges, specialTokenMap)
    }
}
