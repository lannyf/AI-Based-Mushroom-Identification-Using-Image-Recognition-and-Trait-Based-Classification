package se.mushroomid.service;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.Mock;
import org.mockito.junit.jupiter.MockitoExtension;
import org.springframework.http.HttpStatus;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Mono;
import se.mushroomid.model.Step2AnswerRequest;
import se.mushroomid.model.Step2StartRequest;
import se.mushroomid.model.Step3CompareRequest;
import se.mushroomid.model.Step4FinalizeRequest;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.web.reactive.function.client.WebClientResponseException;

import java.util.Map;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.*;

import org.mockito.quality.Strictness;
import org.mockito.junit.jupiter.MockitoSettings;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
@SuppressWarnings("unchecked")
class PythonApiServiceTest {

    // We need to mock the full WebClient fluent chain:
    //   client.post().uri(...).contentType(...).bodyValue(...) → RequestHeadersSpec
    //   .retrieve().onStatus(...).onStatus(...).bodyToMono(...).block()

    @Mock
    private WebClient webClient;

    @Mock
    private WebClient.RequestBodyUriSpec requestBodyUriSpec;

    @Mock
    private WebClient.RequestBodySpec requestBodySpec;

    @Mock
    private WebClient.RequestHeadersSpec requestHeadersSpec;

    @Mock
    private WebClient.RequestHeadersUriSpec requestHeadersUriSpec;

    @Mock
    private WebClient.ResponseSpec responseSpec;

    private PythonApiService service;

    private final ObjectMapper objectMapper = new ObjectMapper();

    @BeforeEach
    void setUp() {
        service = new PythonApiService(webClient, objectMapper);
    }

    // -----------------------------------------------------------------------
    // Helpers — wire the WebClient mock chain for POST requests
    // -----------------------------------------------------------------------

    private void stubPostChain(Object returnValue) {
        when(webClient.post()).thenReturn(requestBodyUriSpec);
        when(requestBodyUriSpec.uri(anyString())).thenReturn(requestBodySpec);
        when(requestBodySpec.contentType(any())).thenReturn(requestBodySpec);
        when(requestBodySpec.body(any())).thenReturn(requestHeadersSpec);
        when(requestBodySpec.bodyValue(any())).thenReturn(requestHeadersSpec);
        when(requestHeadersSpec.retrieve()).thenReturn(responseSpec);
        when(responseSpec.onStatus(any(), any())).thenReturn(responseSpec);
        when(responseSpec.bodyToMono(Object.class))
                .thenReturn(Mono.just(returnValue));
    }

    private void stubGetChain(Object returnValue) {
        when(webClient.get()).thenReturn(requestHeadersUriSpec);
        when(requestHeadersUriSpec.uri(anyString())).thenReturn(requestHeadersSpec);
        when(requestHeadersSpec.retrieve()).thenReturn(responseSpec);
        when(responseSpec.onStatus(any(), any())).thenReturn(responseSpec);
        when(responseSpec.bodyToMono(Object.class))
                .thenReturn(Mono.just(returnValue));
    }

    // -----------------------------------------------------------------------
    // health()
    // -----------------------------------------------------------------------

    @Test
    void health_returnsPythonResponse() {
        Map<String, Object> pythonResp = Map.of("status", "ok", "model_loaded", true);
        stubGetChain(pythonResp);

        Map<String, Object> result = service.health();

        assertThat(result).containsEntry("status", "ok");
    }

    @Test
    void health_returnsEmptyMapWhenPythonReturnsEmpty() {
        stubGetChain(Map.of());

        Map<String, Object> result = service.health();

        assertThat(result).isEmpty();
    }

    // -----------------------------------------------------------------------
    // step2Start()
    // -----------------------------------------------------------------------

    @Test
    void step2Start_returnsQuestion() {
        Map<String, Object> question = Map.of(
                "status", "question",
                "session_id", "sess-abc",
                "question", "Cap shape?",
                "options", java.util.List.of("Convex", "Flat")
        );
        stubPostChain(question);

        Step2StartRequest request = new Step2StartRequest();
        request.setVisibleTraits(Map.of("cap_color", "red"));

        Map<String, Object> result = service.step2Start(request);

        assertThat(result).containsEntry("status", "question");
        assertThat(result).containsEntry("session_id", "sess-abc");
    }

    @Test
    void step2Start_returnsConclusion() {
        Map<String, Object> conclusion = Map.of(
                "status", "conclusion",
                "session_id", "sess-abc",
                "species", "Flugsvamp",
                "edibility", "toxic"
        );
        stubPostChain(conclusion);

        Step2StartRequest request = new Step2StartRequest();
        request.setVisibleTraits(Map.of("cap_color", "red", "spots", "white"));

        Map<String, Object> result = service.step2Start(request);

        assertThat(result).containsEntry("status", "conclusion");
        assertThat(result).containsEntry("edibility", "toxic");
    }

    // -----------------------------------------------------------------------
    // step2Answer()
    // -----------------------------------------------------------------------

    @Test
    void step2Answer_returnsNextQuestion() {
        Map<String, Object> nextQ = Map.of(
                "status", "question",
                "question", "Gill attachment?",
                "options", java.util.List.of("Free", "Adnate")
        );
        stubPostChain(nextQ);

        Step2AnswerRequest request = new Step2AnswerRequest();
        request.setSessionId("sess-abc");
        request.setAnswer("Convex");

        Map<String, Object> result = service.step2Answer(request);

        assertThat(result).containsEntry("status", "question");
    }

    // -----------------------------------------------------------------------
    // step3Compare()
    // -----------------------------------------------------------------------

    @Test
    void step3Compare_returnsTraitMatchScore() {
        Map<String, Object> compareResult = Map.of(
                "trait_match", 0.85,
                "safety_alert", false
        );
        stubPostChain(compareResult);

        Step3CompareRequest request = new Step3CompareRequest();
        request.setSwedishName("Kantarell");
        request.setVisibleTraits(Map.of("cap_color", "yellow"));

        Map<String, Object> result = service.step3Compare(request);

        assertThat(result).containsEntry("trait_match", 0.85);
        assertThat(result).containsEntry("safety_alert", false);
    }

    // -----------------------------------------------------------------------
    // step4Finalize()
    // -----------------------------------------------------------------------

    @Test
    void step4Finalize_returnsFinalResult() {
        Map<String, Object> finalResult = Map.of(
                "verdict", "edible",
                "method_agreement", "full"
        );
        stubPostChain(finalResult);

        Step4FinalizeRequest request = new Step4FinalizeRequest();
        request.setStep1Result(Map.of("top_prediction", "Kantarell"));
        request.setStep2Result(Map.of("status", "conclusion", "species", "Kantarell"));
        request.setStep3Result(Map.of("trait_match", 0.85));

        Map<String, Object> result = service.step4Finalize(request);

        assertThat(result).containsEntry("verdict", "edible");
        assertThat(result).containsEntry("method_agreement", "full");
    }

    @Test
    void step4Finalize_returnsUnknownVerdictForAmbiguousResult() {
        Map<String, Object> finalResult = Map.of(
                "verdict", "unknown",
                "method_agreement", "none"
        );
        stubPostChain(finalResult);

        Step4FinalizeRequest request = new Step4FinalizeRequest();
        request.setStep1Result(Map.of());
        request.setStep2Result(Map.of());
        request.setStep3Result(Map.of());

        Map<String, Object> result = service.step4Finalize(request);

        assertThat(result).containsEntry("verdict", "unknown");
    }
}
