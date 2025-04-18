package musicgen.tokenizer

// tokenizer.json의 구조와 매핑되는 data class

data class TokenizerModel(
    val model: Model,
    val added_tokens: List<AddedToken>,
    val normalizer: Normalizer?,
    val pre_tokenizer: PreTokenizer?,
    val post_processor: PostProcessor?
)

data class Model(
    val type: String,
    val vocab: List<List<Any>> // [token, logprob]
)

data class AddedToken(
    val id: Int,
    val content: String,
    val single_word: Boolean,
    val lstrip: Boolean,
    val rstrip: Boolean,
    val normalized: Boolean,
    val special: Boolean
)

data class Normalizer(
    val type: String?,
    val precompiled_charsmap: String?
)

data class PreTokenizer(val type: String?)
data class PostProcessor(val type: String?)