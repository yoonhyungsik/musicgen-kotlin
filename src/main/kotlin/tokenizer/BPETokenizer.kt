package musicgen.tokenizer

import kotlin.math.min

class BpeTokenizer(
    private val vocab: Map<String, Int>,
    private val merges: List<Pair<String, String>>,
    private val bosToken: String = "",
    private val eosToken: String = "",
    private val padToken: String = "",
    private val unkToken: String = ""
) {
    private val bpeRanks = merges.mapIndexed { idx, pair -> pair to idx }.toMap()

    fun tokenizeToIds(text: String): List<Int> {
        val tokens = mutableListOf<Int>()
        if (bosToken.isNotBlank()) {
            tokens.add(vocab[bosToken] ?: error("Missing BOS token ($bosToken) in vocab"))
        }

        val words = text.lowercase()
            .replace(Regex("[^a-z0-9' ]"), " ")
            .split(Regex("\\s+")).filter { it.isNotEmpty() }

        for (word in words) {
            val bpeTokens = bpe(word)
            for (sub in bpeTokens) {
                tokens.add(vocab[sub] ?: (vocab[unkToken] ?: error("Missing <unk> token")))
            }
        }

        if (eosToken.isNotBlank()) {
            tokens.add(vocab[eosToken] ?: error("Missing EOS token ($eosToken) in vocab"))
        }
        return tokens
    }

    fun buildAttentionMask(length: Int): List<Int> = List(length) { 1 }

    private fun bpe(token: String): List<String> {
        var symbols = token.map { it.toString() }.toMutableList()
        var pairs = getPairs(symbols)

        while (true) {
            val bigram = pairs.minByOrNull { bpeRanks[it] ?: Int.MAX_VALUE } ?: break
            if (bigram !in bpeRanks) break

            val newSymbols = mutableListOf<String>()
            var i = 0
            while (i < symbols.size) {
                val j = min(i + 1, symbols.size - 1)
                if (i < symbols.size - 1 && symbols[i] == bigram.first && symbols[i + 1] == bigram.second) {
                    newSymbols.add(symbols[i] + symbols[i + 1])
                    i += 2
                } else {
                    newSymbols.add(symbols[i])
                    i += 1
                }
            }
            symbols = newSymbols
            pairs = getPairs(symbols)
        }

        return symbols
    }

    private fun getPairs(symbols: List<String>): Set<Pair<String, String>> {
        val pairs = mutableSetOf<Pair<String, String>>()
        for (i in 0 until symbols.size - 1) {
            pairs.add(Pair(symbols[i], symbols[i + 1]))
        }
        return pairs
    }
}