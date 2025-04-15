plugins {
    kotlin("jvm") version "2.1.10"
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(kotlin("test"))
    implementation("com.microsoft.onnxruntime:onnxruntime:1.17.0")
    implementation("org.json:json:20230227")
}

tasks.test {
    useJUnitPlatform()
}
kotlin {
    jvmToolchain(21)
}