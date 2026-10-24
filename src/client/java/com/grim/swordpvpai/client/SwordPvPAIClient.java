package com.grim.swordpvpai.client;

import com.grim.swordpvpai.client.observation.ObservationBridge;
import com.grim.swordpvpai.client.observation.ObservationFeature;
import net.fabricmc.api.ClientModInitializer;

public class SwordPvPAIClient implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        ObservationBridge.start();
        ObservationFeature.register();
    }
}
