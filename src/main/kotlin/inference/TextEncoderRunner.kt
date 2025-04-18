package musicgen.inference

import ai.onnxruntime.*
import musicgen.tokenizer.TokenizerRunner
import java.nio.LongBuffer

class TextEncoderRunner(
    private val session: OrtSession,
    private val tokenizer: TokenizerRunner,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
) {
    fun encode(prompt: String): Array<Array<FloatArray>> {
        val (inputIds, attentionMask) = tokenizer.encodeWithMask(prompt)
        val shape = longArrayOf(1, inputIds.size.toLong())

        val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
        val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)

        val result = session.run(mapOf(
            "input_ids" to inputTensor,
            "attention_mask" to maskTensor
        ))

        return result[0].value as Array<Array<FloatArray>>
    }
}
