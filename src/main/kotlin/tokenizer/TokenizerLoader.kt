package musicgen.tokenizer

import com.google.gson.Gson
import java.io.File

object TokenizerLoader {

    fun loadVocabFromFile(path: String): Map<String, Int> {
        val jsonStr = File(path).readText(Charsets.UTF_8)
        val gson = Gson()
        val mapType = object : com.google.gson.reflect.TypeToken<Map<String, Int>>() {}.type
        return gson.fromJson(jsonStr, mapType)
    }

    fun loadMergesFromFile(path: String): List<Pair<String, String>> {
        return File(path).readLines()
            .filter { it.isNotBlank() && !it.startsWith("#") }
            .map {
                val parts = it.split(" ")
                parts[0] to parts[1]
            }
    }
}