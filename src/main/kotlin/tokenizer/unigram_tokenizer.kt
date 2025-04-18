package musicgen.tokenizer

// === ScorePath: Viterbi 경로 저장용 ===
data class ScorePath(val score: Float, val path: List<Int>)


object UnigramTokenizer {

    // 입력 텍스트를 Viterbi 알고리즘으로 분할하여 토큰 ID 리스트 반환
    fun tokenize(text: String, vocab: Map<String, TokenEntry>): List<Int> {
        val length = text.length
        val best = Array<ScorePath?>(length + 1) { null }
        best[0] = ScorePath(0f, emptyList())

        for (i in 1..length) {
            for (j in 0 until i) {
                val substring = text.substring(j, i)
                val entry = vocab[substring] ?: continue
                val prev = best[j] ?: continue

                val newScore = prev.score + entry.score
                val newPath = prev.path + entry.id

                if (best[i] == null || newScore > best[i]!!.score) {
                    best[i] = ScorePath(newScore, newPath)
                }
            }
        }

        return best[length]?.path ?: emptyList()
    }

    // vocab 리스트를 map으로 변환: token -> TokenEntry
    fun buildVocabMap(vocabList: List<List<Any>>): Map<String, TokenEntry> {
        val map = mutableMapOf<String, TokenEntry>()
        vocabList.forEachIndexed { idx, entry ->
            val token = entry[0] as String
            val score = (entry[1] as Number).toFloat()
            map[token] = TokenEntry(token, idx, score)
        }
        return map
    }
}