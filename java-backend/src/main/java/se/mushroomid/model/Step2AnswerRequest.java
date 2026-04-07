package se.mushroomid.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

/** Request body for POST /api/identify/step2/answer */
@Data
public class Step2AnswerRequest {

    @JsonProperty("session_id")
    private String sessionId;

    /** Must exactly match one of the options returned by step2/start or previous step2/answer. */
    private String answer;
}
