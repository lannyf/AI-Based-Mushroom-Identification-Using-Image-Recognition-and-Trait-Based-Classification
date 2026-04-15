package se.mushroomid.service;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.http.HttpStatusCode;
import org.springframework.http.MediaType;
import org.springframework.http.client.MultipartBodyBuilder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.reactive.function.BodyInserters;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Mono;
import se.mushroomid.model.Step2AnswerRequest;
import se.mushroomid.model.Step2StartRequest;
import se.mushroomid.model.Step3CompareRequest;
import se.mushroomid.model.Step4FinalizeRequest;

import java.io.IOException;
import java.util.Map;

/**
 * Proxy service that forwards every request from Flutter to the Python FastAPI
 * pipeline and returns the raw JSON response unchanged.
 *
 * Python endpoints called:
 *   POST /identify                   — Step 1 (image + traits)
 *   POST /identify/Species_tree_traversal/start       — Step 2 begin traversal
 *   POST /identify/Species_tree_traversal/answer      — Step 2 continue traversal
 *   GET  /identify/Species_tree_traversal/session/{id}— Step 2 session state
 *   POST /identify/comparison/compare     — Step 3 trait comparison
 *   POST /identify/prediction/finalize    — Step 4 final aggregation
 *   GET  /health                     — Python health probe
 */
@Service
public class PythonApiService {

    private final WebClient client;
    private final ObjectMapper objectMapper;

    public PythonApiService(@Qualifier("pythonApiClient") WebClient client,
                            ObjectMapper objectMapper) {
        this.client = client;
        this.objectMapper = objectMapper;
    }

    // -----------------------------------------------------------------------
    // Step 1 — image upload + visual analysis
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public Map<String, Object> identify(MultipartFile image, Map<String, Object> traits) {
        MultipartBodyBuilder builder = new MultipartBodyBuilder();

        try {
            byte[] bytes = image.getBytes();
            String filename = image.getOriginalFilename() != null
                    ? image.getOriginalFilename() : "upload.jpg";

            String contentTypeStr = image.getContentType();
            MediaType mediaType = (contentTypeStr != null && !contentTypeStr.isBlank())
                    ? MediaType.parseMediaType(contentTypeStr)
                    : MediaType.APPLICATION_OCTET_STREAM;

            builder.part("image", new ByteArrayResource(bytes) {
                @Override public String getFilename() { return filename; }
            }).contentType(mediaType);

            builder.part("traits", objectMapper.writeValueAsString(traits));
        } catch (IOException e) {
            throw new RuntimeException("Failed to read uploaded image", e);
        }

        return (Map<String, Object>) callPython(
                client.post()
                        .uri("/identify")
                        .contentType(MediaType.MULTIPART_FORM_DATA)
                        .body(BodyInserters.fromMultipartData(builder.build()))
        );
    }

    // -----------------------------------------------------------------------
    // Step 2 — species key tree traversal
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public Map<String, Object> step2Start(Step2StartRequest request) {
        return (Map<String, Object>) callPython(
                client.post()
                        .uri("/identify/Species_tree_traversal/start")
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(request)
        );
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> step2Answer(Step2AnswerRequest request) {
        return (Map<String, Object>) callPython(
                client.post()
                        .uri("/identify/Species_tree_traversal/answer")
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(request)
        );
    }

    @SuppressWarnings("unchecked")
    public Map<String, Object> step2SessionState(String sessionId) {
        return (Map<String, Object>) callPython(
                client.get()
                        .uri("/identify/Species_tree_traversal/session/{id}", sessionId)
        );
    }

    // -----------------------------------------------------------------------
    // Step 3 — trait database comparison
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public Map<String, Object> step3Compare(Step3CompareRequest request) {
        return (Map<String, Object>) callPython(
                client.post()
                        .uri("/identify/comparison/compare")
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(request)
        );
    }

    // -----------------------------------------------------------------------
    // Step 4 — final aggregation
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public Map<String, Object> step4Finalize(Step4FinalizeRequest request) {
        return (Map<String, Object>) callPython(
                client.post()
                        .uri("/identify/prediction/finalize")
                        .contentType(MediaType.APPLICATION_JSON)
                        .bodyValue(request)
        );
    }

    // -----------------------------------------------------------------------
    // Health
    // -----------------------------------------------------------------------

    @SuppressWarnings("unchecked")
    public Map<String, Object> health() {
        return (Map<String, Object>) callPython(client.get().uri("/health"));
    }

    // -----------------------------------------------------------------------
    // Internal helper
    // -----------------------------------------------------------------------

    private Object callPython(WebClient.RequestHeadersSpec<?> spec) {
        return spec
                .retrieve()
                .onStatus(HttpStatusCode::is4xxClientError, response ->
                        response.bodyToMono(String.class).flatMap(body ->
                                Mono.error(new ResponseStatusException(response.statusCode(), body))
                        )
                )
                .onStatus(HttpStatusCode::is5xxServerError, response ->
                        response.bodyToMono(String.class).flatMap(body ->
                                Mono.error(new ResponseStatusException(response.statusCode(),
                                        "Python backend error: " + body))
                        )
                )
                .bodyToMono(Object.class)
                .block();
    }
}
