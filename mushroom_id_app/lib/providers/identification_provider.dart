import 'dart:typed_data';
import 'package:get/get.dart';
import 'package:logger/logger.dart';

import '../services/identification_api_service.dart';

/// GetX controller for managing the full 4-step identification flow.
///
/// Step 1: image upload  → step1Result stored
/// Step 2: tree traversal → step2SessionId, step2Question, step2Options
///          user answers until step2Concluded == true (step2Result stored)
/// Step 3: trait comparison → step3Result stored
/// Step 4: final aggregation → step4Result stored  ← this is what ResultsPage shows
class IdentificationProvider extends GetxController {
  final Logger _logger = Logger();
  final IdentificationApiService _api;

  IdentificationProvider({IdentificationApiService? api})
      : _api = api ?? IdentificationApiService();

  // ---------------------------------------------------------------------------
  // Image state
  // ---------------------------------------------------------------------------
  final RxnString selectedImagePath = RxnString();
  final Rx<Uint8List?> selectedImageBytes = Rx<Uint8List?>(null);

  // ---------------------------------------------------------------------------
  // Legacy trait questionnaire (still used if user skips tree traversal)
  // ---------------------------------------------------------------------------
  final RxMap<String, dynamic> selectedTraits = RxMap<String, dynamic>({});

  // ---------------------------------------------------------------------------
  // Pipeline step results
  // ---------------------------------------------------------------------------
  final Rx<Map<String, dynamic>?> step1Result = Rx(null);
  final Rx<Map<String, dynamic>?> step2Result = Rx(null);
  final Rx<Map<String, dynamic>?> step3Result = Rx(null);
  final Rx<Map<String, dynamic>?> step4Result = Rx(null);

  // ---------------------------------------------------------------------------
  // Step 2 session state (tree traversal Q&A)
  // ---------------------------------------------------------------------------
  final RxnString step2SessionId      = RxnString();
  final RxnString step2Question       = RxnString();
  final RxList<String> step2Options   = RxList<String>();
  final RxBool step2Concluded         = false.obs;
  final RxInt step2AutoAnswers        = 0.obs;
  final RxInt step2UserAnswers        = 0.obs;

  // ---------------------------------------------------------------------------
  // UI state
  // ---------------------------------------------------------------------------
  final RxBool isProcessing  = false.obs;
  final RxnString errorMessage = RxnString();
  final RxInt currentStep    = 0.obs; // 0=camera, 1=traversal, 2=results

  // ---------------------------------------------------------------------------
  // Trait options (kept for manual questionnaire fallback)
  // ---------------------------------------------------------------------------
  late final List<String> capShapes;
  late final List<String> colors;
  late final List<String> gillTypes;
  late final List<String> stemTypes;
  late final List<String> habitats;
  late final List<String> seasons;

  @override
  void onInit() {
    super.onInit();
    _initializeTraitOptions();
  }

  void _initializeTraitOptions() {
    capShapes = ['Convex', 'Flat', 'Conical', 'Wavy', 'Bell-shaped', 'Umbrella-shaped'];
    colors    = ['White', 'Beige', 'Brown', 'Red', 'Orange', 'Yellow', 'Green', 'Purple', 'Black', 'Gray'];
    gillTypes = ['Free', 'Attached', 'Decurrent', 'Subdecurrent', 'Crowded', 'Sparse', 'Forked'];
    stemTypes = ['Solid', 'Hollow', 'Fibrous', 'Bulbous', 'Rooted', 'Ring/Annulus', 'Cup/Volva'];
    habitats  = ['Forest', 'Grassland', 'Garden', 'Dead wood', 'Living tree', 'Underground', 'Disturbed soil'];
    seasons   = ['Spring', 'Summer', 'Autumn', 'Winter'];
  }

  // ---------------------------------------------------------------------------
  // Image helpers
  // ---------------------------------------------------------------------------

  void setSelectedImage({required String imagePath, Uint8List? imageBytes}) {
    selectedImagePath.value = imagePath;
    selectedImageBytes.value = imageBytes;
    _logger.i('Image selected: $imagePath');
  }

  void clearImage() {
    selectedImagePath.value = null;
    selectedImageBytes.value = null;
    reset();
  }

  // ---------------------------------------------------------------------------
  // Step 1 — run visual analysis on image
  // ---------------------------------------------------------------------------

  Future<void> runStep1() async {
    final imagePath = selectedImagePath.value;
    if (imagePath == null) {
      setError('No image selected');
      return;
    }
    setProcessing(true);
    clearError();
    try {
      final result = await _api.identifyStep1(
        imagePath: imagePath,
        imageBytes: selectedImageBytes.value,
        traits: Map.from(selectedTraits),
      );
      step1Result.value = result;
      _logger.i('Step 1 complete: ${result['top_prediction']}');
    } catch (e) {
      setError(e.toString());
    } finally {
      setProcessing(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Step 2 — start tree traversal
  // ---------------------------------------------------------------------------

  Future<void> runStep2Start() async {
    final s1 = step1Result.value;
    if (s1 == null) { setError('Run Step 1 first'); return; }

    final step1Map = s1['trait_extraction'] as Map?;
    final rawVisibleTraits = step1Map?['visible_traits'] as Map?;
    final visibleTraits = rawVisibleTraits != null
        ? Map<String, dynamic>.from(rawVisibleTraits)
        : <String, dynamic>{};
    setProcessing(true);
    clearError();
    try {
      final result = await _api.step2Start(visibleTraits: visibleTraits);
      _applyStep2Response(result);
    } catch (e) {
      setError(e.toString());
    } finally {
      setProcessing(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Step 2 — answer a question
  // ---------------------------------------------------------------------------

  Future<void> runStep2Answer(String answer) async {
    final sid = step2SessionId.value;
    if (sid == null) { setError('No active Step 2 session'); return; }

    setProcessing(true);
    clearError();
    try {
      final result = await _api.step2Answer(sessionId: sid, answer: answer);
      _applyStep2Response(result);
    } catch (e) {
      setError(e.toString());
    } finally {
      setProcessing(false);
    }
  }

  void _applyStep2Response(Map<String, dynamic> result) {
    step2SessionId.value  = result['session_id'] as String?;

    final autoAnswers = (result['auto_answers'] as num?)?.toInt();
    final userAnswers = (result['user_answers'] as num?)?.toInt();
    final autoAnswered = result['auto_answered'] as List?;
    final path = result['path'] as List?;

    final derivedAutoAnswers = autoAnswered?.length ?? 0;
    final derivedUserAnswers =
        path == null ? 0 : (path.length - derivedAutoAnswers).clamp(0, path.length);

    step2AutoAnswers.value = autoAnswers ?? derivedAutoAnswers;
    step2UserAnswers.value = userAnswers ?? derivedUserAnswers;

    if (result['status'] == 'conclusion') {
      step2Result.value = result;
      step2Concluded.value = true;
      step2Question.value = null;
      step2Options.clear();
      _logger.i('Step 2 concluded: ${result['species']}');
    } else {
      step2Question.value = result['question'] as String?;
      step2Options.value  = List<String>.from(result['options'] as List? ?? []);
      step2Concluded.value = false;
    }
  }

  // ---------------------------------------------------------------------------
  // Step 3 — trait comparison
  // ---------------------------------------------------------------------------

  Future<void> runStep3() async {
    final s1 = step1Result.value;
    final s2 = step2Result.value;
    if (s1 == null || s2 == null) { setError('Steps 1 and 2 must be complete'); return; }

    final swedishName = s2['species'] as String? ?? '';
    final step1Map = s1['trait_extraction'] as Map?;
    final rawVisibleTraits = step1Map?['visible_traits'] as Map?;
    final visibleTraits = rawVisibleTraits != null
        ? Map<String, dynamic>.from(rawVisibleTraits)
        : <String, dynamic>{};

    setProcessing(true);
    clearError();
    try {
      final result = await _api.step3Compare(
        swedishName: swedishName,
        visibleTraits: visibleTraits,
      );
      step3Result.value = result;
      _logger.i('Step 3 complete: trait score ${result['trait_match']?['score']}');
    } catch (e) {
      setError(e.toString());
    } finally {
      setProcessing(false);
    }
  }

  // ---------------------------------------------------------------------------
  // Step 4 — final aggregation
  // ---------------------------------------------------------------------------

  Future<void> runStep4() async {
    final s1 = step1Result.value;
    final s2 = step2Result.value;
    final s3 = step3Result.value;
    if (s1 == null || s2 == null || s3 == null) {
      setError('All three steps must be complete before finalising');
      return;
    }
    setProcessing(true);
    clearError();
    try {
      final result = await _api.step4Finalize(
        step1Result: s1,
        step2Result: s2,
        step3Result: s3,
      );
      step4Result.value = result;
      _logger.i('Step 4 complete: ${result['final_recommendation']?['swedish_name']}');
    } catch (e) {
      setError(e.toString());
    } finally {
      setProcessing(false);
    }
  }

  /// Convenience: runs Steps 3 and 4 sequentially after Step 2 concludes.
  Future<void> runStep3AndStep4() async {
    await runStep3();
    if (errorMessage.value != null) return;
    await runStep4();
  }

  // ---------------------------------------------------------------------------
  // UI helpers
  // ---------------------------------------------------------------------------

  void setTrait(String category, dynamic value) => selectedTraits[category] = value;
  void setTraits(Map<String, dynamic> traits) => selectedTraits.addAll(traits);
  void clearTraits() => selectedTraits.clear();

  void setProcessing(bool value) => isProcessing.value = value;
  void setError(String? error) {
    errorMessage.value = error;
    if (error != null) _logger.e('Error: $error');
  }
  void clearError() => errorMessage.value = null;

  void nextStep() { if (currentStep.value < 2) currentStep.value++; }
  void previousStep() { if (currentStep.value > 0) currentStep.value--; }
  void goToStep(int step) { if (step >= 0 && step <= 2) currentStep.value = step; }

  List<String> getTraitOptions(String category) {
    switch (category.toLowerCase()) {
      case 'cap_shape': case 'capshape': return capShapes;
      case 'color':     return colors;
      case 'gill_type': case 'gilltype': return gillTypes;
      case 'stem_type': case 'stemtype': return stemTypes;
      case 'habitat':   return habitats;
      case 'season':    return seasons;
      default: return [];
    }
  }

  /// Reset all pipeline state (but keep the selected image).
  void reset() {
    step1Result.value = null;
    step2Result.value = null;
    step3Result.value = null;
    step4Result.value = null;
    step2SessionId.value = null;
    step2Question.value  = null;
    step2Options.clear();
    step2Concluded.value = false;
    step2AutoAnswers.value = 0;
    step2UserAnswers.value = 0;
    selectedTraits.clear();
    isProcessing.value = false;
    errorMessage.value = null;
    currentStep.value  = 0;
  }

  Map<String, dynamic> getIdentificationData() => {
    'imagePath':  selectedImagePath.value,
    'imageBytes': selectedImageBytes.value,
    'traits':     Map.from(selectedTraits),
  };

  bool validateTraitsCompleted() {
    final required = ['cap_shape', 'color', 'gill_type', 'stem_type', 'habitat', 'season'];
    final missing  = required.where((t) => !selectedTraits.containsKey(t)).toList();
    if (missing.isNotEmpty) { setError('Missing traits: ${missing.join(", ")}'); return false; }
    return true;
  }
}
