package musicgen.tokenizer

object TokenizerAppInit {
    fun buildTokenizerFromJson(jsonPath: String): TokenizerRunner {
        val (vocab, merges, special) = TokenizerJsonLoader.loadFromTokenizerJson(jsonPath)
        val bos = special["<s>"]?.let { "<s>" } ?: ""  // 없으면 빈 문자열
        val eos = special["</s>"]?.let { "</s>" } ?: ""
        val pad = special["<pad>"]?.let { "<pad>" } ?: ""
        val unk = special["<unk>"]?.let { "<unk>" } ?: ""

        val tokenizer = BpeTokenizer(
            vocab = vocab,
            merges = merges,
            bosToken = bos,
            eosToken = eos,
            padToken = pad,
            unkToken = unk
        )


        return TokenizerRunner(tokenizer)
    }

    fun getPadTokenId(jsonPath: String): Int {
        val (_, _, special) = TokenizerJsonLoader.loadFromTokenizerJson(jsonPath)
        return special["<pad>"] ?: error("<pad> token not found")
    }
}
