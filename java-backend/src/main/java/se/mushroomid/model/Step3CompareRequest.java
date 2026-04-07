package se.mushroomid.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

/** Request body for POST /api/identify/step3/compare */
@Data
public class Step3CompareRequest {

    /** Swedish species name from Step 2 conclusion. */
    @JsonProperty("swedish_name")
    private String swedishName;

    /** Visible traits dict from Step 1 (visible_traits field). */
    @JsonProperty("visible_traits")
    private Map<String, Object> visibleTraits;
}
