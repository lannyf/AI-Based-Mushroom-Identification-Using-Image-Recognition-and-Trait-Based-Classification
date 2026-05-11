# Rewritten Method Section (aligned with current codebase)

Use the following as a complete replacement for your current `\section{Method}`.

---

```latex
\section{Method}
\label{sec:method}

Section~\ref{sec:background} established why mushroom identification benefits from several complementary reasoning channels. This section describes how the prototype application was implemented and how its current state was analysed.

\subsection{Dataset Construction}
\label{sec:method-data}

The structured dataset was derived from \emph{Nya svampboken} and complemented by image metadata and downloaded mushroom photographs~\cite{Nya_svampboken}. The raw data layer contains seven files under \texttt{data/raw}:

\begin{itemize}
\item \texttt{species.csv} --- species names, edibility flags, toxicity levels, and scientific names for 50 species;
\item \texttt{species\_traits.csv} and \texttt{species\_traits.xml} --- morphological trait records (1,049 rows in CSV, plus an XML representation grouped by trait category);
\item \texttt{species\_images.csv} and \texttt{dataset\_split.csv} --- image references and train/validation/test assignments;
\item \texttt{lookalikes.csv} --- eight documented edible-vs-toxic confusion pairs with distinguishing-feature text and a \textsc{high} or \textsc{critical} confusion-likelihood label;
\item \texttt{key.xml} --- an XML-encoded polytomous identification key in Swedish.
\end{itemize}

The training data for the CNN were collected from iNaturalist. Observations were filtered to mushroom taxa in Sweden, with Scandinavia as a fallback when Swedish samples were scarce. The resulting curated image subset contains 210 photographs distributed across the seven species handled by the neural classifier. An additional 142 photographs of five other species are held in the raw image directory for future expansion but are not used by the current classifier.

The project includes Python utilities for loading, validating, and exporting the dataset. \texttt{data/dataset\_utils.py} provides the main loader and validation routines, while \texttt{data/prepare\_data.py} converts the raw material into processed, machine-readable formats.

\subsection{System Architecture}
\label{sec:method-architecture}

The implemented application uses a layered architecture with a Flutter client, a Java Spring Boot proxy backend, and a Python FastAPI identification backend. The Flutter client provides the interface for image capture, manual trait input, key-based follow-up questions, result display, settings, and local history. The Java layer exposes a stable API for the client and forwards requests to the Python backend. The Python backend contains the actual identification logic.

The Python backend implements a four-step pipeline:

\begin{enumerate}
\item \textbf{Visual trait extraction:} classical computer-vision analysis of an uploaded photograph, producing a \texttt{visible\_traits} structure and an optional CNN \texttt{ml\_prediction}. No fused prediction is produced at this stage.
\item \textbf{Species tree traversal:} traversal of the decision key derived from \texttt{key.xml}, with automatic answers where image-derived traits suffice and optional user-supplied \texttt{pre\_answers} where they do not;
\item \textbf{Comparison:} validation of the candidate species against the trait database and the eight documented lookalike records;
\item \textbf{Final aggregation:} weighted combination of the image, tree, and database signals into a structured result with confidence and safety information.
\end{enumerate}

\begin{figure}[h]
    \centering
    \includegraphics[width=1\textwidth]{Kanduppsats/img/sysdiagram.png}
    \caption{System Architecture diagram}
    \label{Kanduppsats/img/Systemarc.png}
\end{figure}

\subsection{Implementation of the Identification Methods}
\label{sec:method-models}

The project implements three AI-oriented identification methods together with supporting rule-based and aggregation logic.

\subsubsection{Image-Oriented Identification}
\label{sec:method-image}

The image-oriented path contains two components: a convolutional neural network and a classical computer-vision trait extractor. Both operate on the same uploaded photograph but produce different outputs.

\paragraph{Convolutional Neural Network.}
The CNN is built on an EfficientNet-B3 backbone~\cite{Tan2019} using the PyTorch Image Models (\texttt{timm}) library. The model was initialised with ImageNet-pretrained weights and fine-tuned on a curated subset of seven mushroom species: \textit{Fly Agaric}, \textit{Chanterelle}, \textit{False Chanterelle}, \textit{Porcini}, \textit{Other Boletus}, \textit{Amanita virosa}, and \textit{Black Trumpet}. The input size is $300 \times 300$ pixels, and training followed a two-phase transfer-learning schedule: the classification head was trained first while the backbone remained frozen, after which the full network was fine-tuned end-to-end at a reduced learning rate.

At inference time, the CNN outputs a softmax distribution over the seven classes. The top prediction and its confidence are packaged as an \texttt{ml\_prediction} structure. This structure is not a final decision; it is passed forward as one of three evidence signals in the aggregation step.

\paragraph{Classical Visual Trait Extractor.}
Alongside the CNN, a classical computer-vision extractor analyses the photograph using hand-crafted algorithms. It computes dominant and secondary colours from hue histograms, estimates cap shape via contour analysis, measures surface texture through edge-density and local-binary-pattern statistics, detects ridge-like structures, and assesses overall brightness. These measurements are returned in a \texttt{visible\_traits} dictionary. The colour analysis also produces ratio statistics (red, orange-yellow, brown, white, dark) that are returned to the client for diagnostic display.

The classical extractor serves two purposes. First, it produces interpretable evidence that can be inspected directly. Second, its output drives the automatic answering in the key-traversal step and supplies the trait profile for database comparison. If the CNN weights are unavailable, the extractor still returns \texttt{visible\_traits}, ensuring that the remainder of the pipeline can operate without the neural component.

\paragraph{Optional Segmentation Preprocessing.}
The project includes a YOLOv8 segmentation model that can isolate the mushroom fruiting body from background pixels. When segmentation is enabled and the mask quality is sufficient, the visual trait extractor restricts its analysis to masked pixels only. This reduces contamination from soil, vegetation, and shadows. If segmentation is unavailable or the mask quality is low, the system falls back to full-image analysis.

\subsubsection{Trait-Based Identification}
\label{sec:method-traits}

Trait-based reasoning in the project follows two complementary mechanisms: an expert identification-key traversal and a trait-database comparison.

\paragraph{Identification-Key Traversal.}
The key is stored as an XML-encoded polytomous tree derived from a Swedish field guide. Each internal node poses a diagnostic question (for example, \emph{``Hur ser svampen ut?''}), and each branch corresponds to a possible answer. Traversing the tree excludes taxa that are incompatible with the observed character states.

The traversal engine, \texttt{KeyTreeEngine}, attempts to answer each question automatically from the \texttt{visible\_traits} produced by the image extractor. For example, if the extractor reports a brown-dominated cap and a pore-bearing underside, the engine selects the branch for bolete-like mushrooms without asking the user. When the image traits are ambiguous or insufficient, the engine returns the current question and its options to the client.

The engine also accepts an optional \texttt{ml\_hint} from the CNN. If the CNN predicts a species with high confidence, and that species has a known position in the key, the engine uses the hint to guide branch selection. For species not present in the key (for example, \textit{Fly Agaric}), the engine can bypass traversal entirely and return a pre-check conclusion derived from the project's unsupported-species lookup table.

\paragraph{User-Supplied Pre-answers.}
In addition to image-derived traits, the traversal engine accepts \texttt{pre\_answers}: user-supplied responses keyed by the exact question text. These answers are applied \textbf{only} when the image traits cannot provide a conclusive auto-answer. This precedence rule ensures that computer-vision evidence remains primary, while user input acts as a disambiguation mechanism. If a \texttt{pre\_answer} does not match any valid option for the current question, it is silently ignored.

\paragraph{Trait-Database Comparison.}
Once the key traversal reaches a leaf (or a pre-check conclusion), the resulting species name is validated against the trait database. The comparator scores the \texttt{visible\_traits} against the species' stored morphological profile, reporting matched traits, conflicts, and traits that cannot be compared. It also retrieves documented lookalike species from \texttt{lookalikes.csv} and their distinguishing features. If any lookalike is toxic or deadly, a safety alert is raised.

\subsubsection{Language-Model-Based Identification}
\label{sec:method-llm}

The language-model component is implemented as a \textbf{standalone consultation endpoint}, not as a fused input to the main pipeline. It takes the \texttt{visible\_traits} from Step~1, constructs a natural-language observation text, and queries a local Ollama instance running a Llama~3.2~3B model. The endpoint returns a species prediction, confidence estimate, and explanatory reasoning.

The endpoint uses a hard-failure policy: if the Ollama server is unreachable, it returns an HTTP 503 error with no rule-based fallback. This design keeps the LLM as an optional second-opinion channel rather than a hidden dependency of the main identification flow. The observation text uses only image-derived traits; it does not incorporate questionnaire answers or form data.

\subsubsection{Final Aggregation}
\label{sec:method-aggregation}

The project's sole fusion component is the \texttt{FinalAggregator}. It accepts the outputs of Steps 1, 2, and 3 and combines them into a single structured result. The aggregation uses fixed weights:

\begin{itemize}
    \item Step 2 (key traversal conclusion): 45\%;
    \item Step 1 (CNN image analysis): 35\%;
    \item Step 3 (trait database match): 20\%.
\end{itemize}

A $+10\%$ agreement bonus is applied when the CNN top prediction and the key-traversal conclusion agree on the same species. The bonus is capped at 1.0. The aggregator also generates a safety verdict (\textit{edible}, \textit{inedible}, \textit{toxic}, or \textit{unknown}) from the species' toxicity metadata, and it surfaces any lookalike warnings produced by the trait-database comparator.

An earlier \texttt{HybridClassifier} component, which fused CNN, form-trait, and LLM predictions in Step~1 using weighted-average, geometric-mean, and voting strategies, was removed during a pipeline refactor. Its fusion logic was redundant because \texttt{FinalAggregator} already combined the same signals with a more principled weighting scheme, and the early Step~1 prediction was discarded when the client later called the finalisation step.

\subsection{Client and Persistence Design}
\label{sec:method-client}

The Flutter client uses GetX controllers to manage identification state, language selection, and identification history. The client can connect either to the Java proxy or directly to the Python backend, depending on the configured base URL. For persistence, the application uses SharedPreferences on the web and SQLite on non-web platforms. This design makes it possible to support both browser-based demonstrations and mobile-oriented usage with a single code base.

\subsection{Evaluation Procedure}
\label{sec:method-evaluation}

The thesis evaluates the current project state through project artefacts and existing verification routines rather than through a new large-scale field experiment. Three types of evidence were used.

\begin{itemize}
\item \textbf{Implementation artefacts:} source files, architecture documents, and phase summaries were used to identify which planned components had been implemented.
\item \textbf{Automated verification:} Python unit tests, regression tests, and Flutter tests were used as evidence that the implemented components work as intended. The Python test suite covers the visual trait extractor, the key-tree traversal engine (including the \texttt{pre\_answers} mechanism), the trait-database comparator, and the final aggregator.
\item \textbf{Benchmark suite:} standalone method runners for CNN, tree traversal, trait-database comparison, and multimodal fusion were executed on a held-out image set to measure top-k accuracy, coverage, and inference time. These measurements serve as evidence for the thesis hypotheses.
\end{itemize}

This procedure does not replace a full empirical evaluation on a representative field dataset. Instead, it provides a defensible account of the current prototype and a concrete basis for the discussion in Section~\ref{sec:results}.
```
