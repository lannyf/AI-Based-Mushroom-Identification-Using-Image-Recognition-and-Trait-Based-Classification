package se.mushroomid.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

/** Request body for POST /api/identify/step4/finalize */
@Data
public class Step4FinalizeRequest {

    /** Full /api/identify response (contains the "step1" key). */
    @JsonProperty("step1_result")
    private Map<String, Object> step1Result;

    /** Conclusion response from step2/start or step2/answer (status == "conclusion"). */
    @JsonProperty("step2_result")
    private Map<String, Object> step2Result;

    /** Response from step3/compare. */
    @JsonProperty("step3_result")
    private Map<String, Object> step3Result;
}
