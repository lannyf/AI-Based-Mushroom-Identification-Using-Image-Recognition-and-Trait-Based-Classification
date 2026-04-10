package se.mushroomid.controller;

import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.boot.test.mock.mockito.MockBean;
import org.springframework.http.MediaType;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.web.servlet.MockMvc;
import se.mushroomid.model.Step2AnswerRequest;
import se.mushroomid.model.Step2StartRequest;
import se.mushroomid.model.Step3CompareRequest;
import se.mushroomid.model.Step4FinalizeRequest;
import se.mushroomid.service.PythonApiService;

import java.util.Map;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.*;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(IdentificationController.class)
class IdentificationControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockBean
    private PythonApiService pythonApiService;

    // -----------------------------------------------------------------------
    // GET /api/health
    // -----------------------------------------------------------------------

    @Test
    void health_returnsOk() throws Exception {
        when(pythonApiService.health())
                .thenReturn(new java.util.HashMap<>(Map.of("status", "ok", "python_backend", "ok")));

        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.java_backend").value("ok"))
                .andExpect(jsonPath("$.python_backend").value("ok"));
    }

    @Test
    void health_addsPythonFields() throws Exception {
        when(pythonApiService.health())
                .thenReturn(new java.util.HashMap<>(Map.of("model_loaded", true, "status", "ok")));

        mockMvc.perform(get("/api/health"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.java_backend").value("ok"))
                .andExpect(jsonPath("$.model_loaded").value(true));
    }

    // -----------------------------------------------------------------------
    // POST /api/identify
    // -----------------------------------------------------------------------

    @Test
    void identify_validImageReturnsResult() throws Exception {
        when(pythonApiService.identify(any(), any()))
                .thenReturn(Map.of("top_prediction", "Amanita muscaria", "confidence", 0.91));

        MockMultipartFile image = new MockMultipartFile(
                "image", "test.jpg", "image/jpeg", "fake-image-bytes".getBytes()
        );

        mockMvc.perform(multipart("/api/identify").file(image))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.top_prediction").value("Amanita muscaria"));
    }

    @Test
    void identify_withTraitsJsonReturnsResult() throws Exception {
        when(pythonApiService.identify(any(), any()))
                .thenReturn(Map.of("top_prediction", "Boletus edulis"));

        MockMultipartFile image = new MockMultipartFile(
                "image", "test.jpg", "image/jpeg", "fake-image-bytes".getBytes()
        );
        MockMultipartFile traits = new MockMultipartFile(
                "traits", "", "text/plain", "{\"cap_color\":\"brown\"}".getBytes()
        );

        mockMvc.perform(multipart("/api/identify").file(image).file(traits))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.top_prediction").value("Boletus edulis"));
    }

    // -----------------------------------------------------------------------
    // POST /api/identify/step2/start
    // -----------------------------------------------------------------------

    @Test
    void step2Start_returnsQuestion() throws Exception {
        when(pythonApiService.step2Start(any(Step2StartRequest.class)))
                .thenReturn(Map.of(
                        "status", "question",
                        "session_id", "sess-001",
                        "question", "What is the cap shape?",
                        "options", java.util.List.of("Convex", "Flat")
                ));

        mockMvc.perform(post("/api/identify/step2/start")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"visible_traits\":{\"cap_color\":\"red\"}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("question"))
                .andExpect(jsonPath("$.session_id").value("sess-001"));
    }

    // -----------------------------------------------------------------------
    // POST /api/identify/step2/answer
    // -----------------------------------------------------------------------

    @Test
    void step2Answer_returnsNextQuestion() throws Exception {
        when(pythonApiService.step2Answer(any(Step2AnswerRequest.class)))
                .thenReturn(Map.of(
                        "status", "question",
                        "session_id", "sess-001",
                        "question", "Are the gills free?",
                        "options", java.util.List.of("Yes", "No")
                ));

        mockMvc.perform(post("/api/identify/step2/answer")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"session_id\":\"sess-001\",\"answer\":\"Convex\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("question"));
    }

    @Test
    void step2Answer_returnsConclusion() throws Exception {
        when(pythonApiService.step2Answer(any(Step2AnswerRequest.class)))
                .thenReturn(Map.of(
                        "status", "conclusion",
                        "session_id", "sess-001",
                        "species", "Flugsvamp",
                        "edibility", "toxic"
                ));

        mockMvc.perform(post("/api/identify/step2/answer")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"session_id\":\"sess-001\",\"answer\":\"Yes\"}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.status").value("conclusion"))
                .andExpect(jsonPath("$.edibility").value("toxic"));
    }

    // -----------------------------------------------------------------------
    // GET /api/identify/step2/session/{sessionId}
    // -----------------------------------------------------------------------

    @Test
    void step2SessionState_returnsState() throws Exception {
        when(pythonApiService.step2SessionState(eq("sess-001")))
                .thenReturn(Map.of("session_id", "sess-001", "depth", 2));

        mockMvc.perform(get("/api/identify/step2/session/sess-001"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.session_id").value("sess-001"))
                .andExpect(jsonPath("$.depth").value(2));
    }

    // -----------------------------------------------------------------------
    // POST /api/identify/step3/compare
    // -----------------------------------------------------------------------

    @Test
    void step3Compare_returnsTraitMatch() throws Exception {
        when(pythonApiService.step3Compare(any(Step3CompareRequest.class)))
                .thenReturn(Map.of(
                        "trait_match", 0.82,
                        "safety_alert", false,
                        "lookalikes", java.util.List.of()
                ));

        mockMvc.perform(post("/api/identify/step3/compare")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content("{\"swedish_name\":\"Kantarell\"," +
                                 "\"visible_traits\":{\"cap_color\":\"yellow\"}}"))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.trait_match").value(0.82))
                .andExpect(jsonPath("$.safety_alert").value(false));
    }

    // -----------------------------------------------------------------------
    // POST /api/identify/step4/finalize
    // -----------------------------------------------------------------------

    @Test
    void step4Finalize_returnsFinalRecommendation() throws Exception {
        when(pythonApiService.step4Finalize(any(Step4FinalizeRequest.class)))
                .thenReturn(Map.of(
                        "verdict", "edible",
                        "final_recommendation", Map.of(
                                "species", "Kantarell",
                                "overall_confidence", 0.88
                        )
                ));

        String body = """
                {
                  "step1_result": {"top_prediction": "Kantarell"},
                  "step2_result": {"status": "conclusion", "species": "Kantarell"},
                  "step3_result": {"trait_match": 0.82}
                }
                """;

        mockMvc.perform(post("/api/identify/step4/finalize")
                        .contentType(MediaType.APPLICATION_JSON)
                        .content(body))
                .andExpect(status().isOk())
                .andExpect(jsonPath("$.verdict").value("edible"))
                .andExpect(jsonPath("$.final_recommendation.species").value("Kantarell"));
    }
}
