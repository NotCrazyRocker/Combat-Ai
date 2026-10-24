# Sword PvP AI

A Fabric mod project for Minecraft 1.21.11.

## Development requirements

- Java 21 or newer
- The included Gradle wrapper

## Run the development client

On Windows:

```powershell
.\gradlew.bat runClient
```

On macOS or Linux:

```sh
./gradlew runClient
```

The first run downloads Minecraft and the required dependencies. The development game directory is `run/`.

## Build

On Windows:

```powershell
.\gradlew.bat build
```

On macOS or Linux:

```sh
./gradlew build
```

The distributable mod JAR is written to `build/libs/`.

## Source layout

- `src/main/java`: common mod initialization and logic
- `src/client/java`: client-only initialization and logic
- `src/main/resources`: mod metadata and common assets
- `src/client/resources`: client-only resources

The common initializer is `com.grim.swordpvpai.SwordPvPAI`; the client initializer is `com.grim.swordpvpai.client.SwordPvPAIClient`.
