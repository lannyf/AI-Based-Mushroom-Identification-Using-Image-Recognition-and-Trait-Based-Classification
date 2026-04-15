package se.mushroomid.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

/** Request body for POST /api/identify/prediction/finalize */
@Data
public class Step4FinalizeRequest {

    /** Full /api/identify response (contains the "trait_extraction" key). */
    @JsonProperty("trait_extraction_result")
    private Map<String, Object> step1Result;

    /** Conclusion response from step2/start or step2/answer (status == "conclusion"). */
    @JsonProperty("Species_tree_traversal_result")
    private Map<String, Object> step2Result;

    /** Response from step3/compare. */
    @JsonProperty("comparison_result")
    private Map<String, Object> step3Result;
}
