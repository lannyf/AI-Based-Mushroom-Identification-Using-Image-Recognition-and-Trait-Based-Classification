package se.mushroomid.controller;

import org.springframework.http.MediaType;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import se.mushroomid.model.*;
import se.mushroomid.service.PythonApiService;

import java.util.Map;

/**
 * REST controller — exposes the full 4-step mushroom identification pipeline
 * to the Flutter client.
 *
 * All routes are prefixed with /api to distinguish from the Python backend.
 *
 * Flow the Flutter app follows:
 *   1. POST /api/identify                    — upload image, get Step 1 result
 *   2. POST /api/identify/Species_tree_traversal/start        — start species tree traversal
 *      (repeat) POST /api/identify/Species_tree_traversal/answer — answer questions until conclusion
 *   3. POST /api/identify/comparison/compare      — compare candidate against trait DB
 *   4. POST /api/identify/prediction/finalize     — get final recommendation
 */
@RestController
@RequestMapping("/api")
public class IdentificationController {

    private final PythonApiService pythonApiService;

    public IdentificationController(PythonApiService pythonApiService) {
        this.pythonApiService = pythonApiService;
    }

    // -----------------------------------------------------------------------
    // Health
    // -----------------------------------------------------------------------

    @GetMapping("/health")
    public Map<String, Object> health() {
        Map<String, Object> pythonHealth = pythonApiService.health();
        pythonHealth.put("java_backend", "ok");
        return pythonHealth;
    }

    // -----------------------------------------------------------------------
    // Step 1 — image visual analysis + ML prediction
    // -----------------------------------------------------------------------

    /**
     * Upload a mushroom image and get:
     *  - step1.ml_prediction  : top ML species guess with confidence
     *  - step1.visible_traits : colour, cap_shape, texture, ridges detected from image
     *  - top_prediction       : aggregated top species
     *  - predictions          : ranked list of candidates
     *  - lookalikes           : preliminary lookalike warnings
     *
     * @param image  JPEG/PNG image of the mushroom (multipart)
     * @param traits Optional JSON string of manual trait selections (default "{}")
     */
    @PostMapping(value = "/identify", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public Map<String, Object> identify(
            @RequestPart("image") MultipartFile image,
            @RequestPart(value = "traits", required = false) String traits
    ) {
        Map<String, Object> parsedTraits = parseTraits(traits);
        return pythonApiService.identify(image, parsedTraits);
    }

    // -----------------------------------------------------------------------
    // Step 2 — species key tree traversal
    // -----------------------------------------------------------------------

    /**
     * Begin Step 2 traversal.  Post the visible_traits from the Step 1 response.
     * The engine auto-answers questions it can resolve from image data;
     * it returns the first unanswerable question or a conclusion.
     *
     * Response shapes:
     *   {"status":"question",   "session_id":..., "question":..., "options":[...]}
     *   {"status":"conclusion", "session_id":..., "species":...,  "edibility":...}
     */
    @PostMapping("/identify/Species_tree_traversal/start")
    public Map<String, Object> step2Start(@RequestBody Step2StartRequest request) {
        return pythonApiService.step2Start(request);
    }

    /**
     * Provide the user's answer to the current key question and continue traversal.
     * answer must exactly match one of the options from the previous call.
     */
    @PostMapping("/identify/Species_tree_traversal/answer")
    public Map<String, Object> step2Answer(@RequestBody Step2AnswerRequest request) {
        return pythonApiService.step2Answer(request);
    }

    /**
     * Retrieve the current state of an active Step 2 session (debugging / polling).
     */
    @GetMapping("/identify/Species_tree_traversal/session/{sessionId}")
    public Map<String, Object> step2SessionState(@PathVariable String sessionId) {
        return pythonApiService.step2SessionState(sessionId);
    }

    // -----------------------------------------------------------------------
    // Step 3 — trait database comparison + lookalike check
    // -----------------------------------------------------------------------

    /**
     * Compare the Step 2 candidate against the species trait database.
     *
     * Post:
     *   swedish_name   — from Step 2 conclusion.species
     *   visible_traits — from Step 1 step1.visible_traits
     *
     * Returns trait_match score, confirmed lookalikes, and safety_alert flag.
     */
    @PostMapping("/identify/comparison/compare")
    public Map<String, Object> step3Compare(@RequestBody Step3CompareRequest request) {
        return pythonApiService.step3Compare(request);
    }

    // -----------------------------------------------------------------------
    // Step 4 — final aggregation and presentation
    // -----------------------------------------------------------------------

    /**
     * Aggregate all three step outputs into the final answer for the user.
     *
     * Post the full responses from steps 1, 2, and 3.
     *
     * Returns:
     *   final_recommendation — best candidate with overall_confidence, reasoning
     *   ml_alternatives      — image-model top-k list
     *   exchangeable_species — confirmed lookalikes with distinguishing features
     *   safety_warnings      — plain-text warnings for toxic/deadly lookalikes
     *   verdict              — "edible" | "inedible" | "toxic" | "unknown"
     *   method_agreement     — "full" | "partial" | "none"
     */
    @PostMapping("/identify/prediction/finalize")
    public Map<String, Object> step4Finalize(@RequestBody Step4FinalizeRequest request) {
        return pythonApiService.step4Finalize(request);
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    private Map<String, Object> parseTraits(String traitsJson) {
        if (traitsJson == null || traitsJson.isBlank()) return Map.of();
        try {
            return new com.fasterxml.jackson.databind.ObjectMapper()
                    .readValue(traitsJson, Map.class);
        } catch (Exception e) {
            return Map.of();
        }
    }
}
