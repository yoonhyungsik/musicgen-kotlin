package musicgen.inference

import ai.onnxruntime.*
import java.nio.LongBuffer

class DecoderRunner(
    private val session: OrtSession,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
) {
    fun decode(inputIds: LongArray, shape: LongArray): LongArray {
        val inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputIds), shape)
        val attentionMask = LongArray(inputIds.size) { 1L }
        val maskTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(attentionMask), shape)

        val output = session.run(mapOf(
            "input_ids" to inputTensor,
            "encoder_attention_mask" to maskTensor
        ))

        return (output[0].value as Array<LongArray>)[0]
    }
}
