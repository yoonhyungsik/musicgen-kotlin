package musicgen.inference

import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer
import java.nio.LongBuffer

class MusicGenGenerator(
    private val decoderModelPath: String,
    private val vocoderModelPath: String,
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
) {
    private val decoderSession: OrtSession = env.createSession(decoderModelPath, OrtSession.SessionOptions())
    private val vocoderSession: OrtSession = env.createSession(vocoderModelPath, OrtSession.SessionOptions())

    fun generateMusic(tokenIds: LongArray, outputPath: String = "generated_music.wav") {
        // 1. Prepare decoder input
        val shape = longArrayOf(1, tokenIds.size.toLong())
        val buffer = LongBuffer.wrap(tokenIds)
        val inputTensor = OnnxTensor.createTensor(env, buffer, shape)

        // 2. Run decoder
        val decoderOutput = decoderSession.run(mapOf("input_ids" to inputTensor))
        val latent = decoderOutput[0].value as Array<Array<FloatArray>>

        // 3. Convert latent to tensor for vocoder
        val flatLatent = latent.flatten().flatMap { it.toList() }.toFloatArray()
        val vocoderShape = longArrayOf(1, flatLatent.size.toLong())
        val vocoderTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(flatLatent), vocoderShape)

        // 4. Run vocoder
        val vocoderOutput = vocoderSession.run(mapOf("input_audio" to vocoderTensor))
        val audio = vocoderOutput[0].value as Array<FloatArray>

        // 5. Save audio to WAV
        saveAsWav(audio[0], outputPath)
        println("✅ Music generated and saved to $outputPath")
    }

    private fun saveAsWav(audio: FloatArray, outputPath: String) {
        val outputFile = File(outputPath)
        val wavStream = FileOutputStream(outputFile)

        val sampleRate = 32000
        val byteRate = 16 * sampleRate / 8
        val audioData = audio.map { (it * Short.MAX_VALUE).toInt().toShort() }.toShortArray()

        val header = ByteArray(44)
        val dataSize = audioData.size * 2

        // RIFF header
        header[0] = 'R'.code.toByte(); header[1] = 'I'.code.toByte(); header[2] = 'F'.code.toByte(); header[3] = 'F'.code.toByte()
        val chunkSize = 36 + dataSize
        writeInt(header, 4, chunkSize)
        header[8] = 'W'.code.toByte(); header[9] = 'A'.code.toByte(); header[10] = 'V'.code.toByte(); header[11] = 'E'.code.toByte()

        // fmt subchunk
        header[12] = 'f'.code.toByte(); header[13] = 'm'.code.toByte(); header[14] = 't'.code.toByte(); header[15] = ' '.code.toByte()
        writeInt(header, 16, 16) // Subchunk1Size
        writeShort(header, 20, 1) // AudioFormat (1 = PCM)
        writeShort(header, 22, 1) // NumChannels
        writeInt(header, 24, sampleRate)
        writeInt(header, 28, byteRate)
        writeShort(header, 32, 2) // BlockAlign
        writeShort(header, 34, 16) // BitsPerSample

        // data subchunk
        header[36] = 'd'.code.toByte(); header[37] = 'a'.code.toByte(); header[38] = 't'.code.toByte(); header[39] = 'a'.code.toByte()
        writeInt(header, 40, dataSize)

        wavStream.write(header)

        val byteBuffer = ByteArray(dataSize)
        var i = 0
        audioData.forEach { sample ->
            byteBuffer[i++] = (sample.toInt() and 0xff).toByte()
            byteBuffer[i++] = ((sample.toInt() shr 8) and 0xff).toByte()
        }

        wavStream.write(byteBuffer)
        wavStream.close()
    }

    private fun writeInt(buffer: ByteArray, offset: Int, value: Int) {
        buffer[offset] = (value and 0xff).toByte()
        buffer[offset + 1] = ((value shr 8) and 0xff).toByte()
        buffer[offset + 2] = ((value shr 16) and 0xff).toByte()
        buffer[offset + 3] = ((value shr 24) and 0xff).toByte()
    }

    private fun writeShort(buffer: ByteArray, offset: Int, value: Int) {
        buffer[offset] = (value and 0xff).toByte()
        buffer[offset + 1] = ((value shr 8) and 0xff).toByte()
    }
}