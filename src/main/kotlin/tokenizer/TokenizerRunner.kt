package musicgen.tokenizer

import musicgen.tokenizer.BpeTokenizer

class TokenizerRunner(
    private val tokenizer: BpeTokenizer
) {
    fun encodeWithMask(text: String): Pair<LongArray, LongArray> {
        val tokenIds = tokenizer.tokenizeToIds(text)
        val attentionMask = tokenizer.buildAttentionMask(tokenIds.size)
        return Pair(
            tokenIds.map { it.toLong() }.toLongArray(),
            attentionMask.map { it.toLong() }.toLongArray()
        )
    }
}
