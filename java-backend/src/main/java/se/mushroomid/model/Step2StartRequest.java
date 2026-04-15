package se.mushroomid.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.Map;

/** Request body for POST /api/identify/Species_tree_traversal/start */
@Data
public class Step2StartRequest {

    @JsonProperty("session_id")
    private String sessionId;

    @JsonProperty("visible_traits")
    private Map<String, Object> visibleTraits;
}
