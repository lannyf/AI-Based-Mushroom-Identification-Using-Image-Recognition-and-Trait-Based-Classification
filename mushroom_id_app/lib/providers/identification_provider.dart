import 'dart:typed_data';
import 'package:get/get.dart';
import 'package:logger/logger.dart';

/// GetX controller for managing identification flow state.
/// 
/// Manages:
/// - Current image being processed
/// - Selected traits from questionnaire
/// - Navigation between pages
/// - Identification results
/// - Error states
class IdentificationProvider extends GetxController {
  final Logger _logger = Logger();

  // Observable state variables
  final RxnString selectedImagePath = RxnString();
  final Rx<Uint8List?> selectedImageBytes = Rx<Uint8List?>(null);
  final RxMap<String, dynamic> selectedTraits = RxMap<String, dynamic>({});
  final RxBool isProcessing = RxBool(false);
  final RxnString errorMessage = RxnString();
  final RxInt currentStep = RxInt(0); // 0: camera, 1: questionnaire, 2: results

  // Trait categories
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

  /// Initializes available trait options
  void _initializeTraitOptions() {
    capShapes = [
      'Convex',
      'Flat',
      'Conical',
      'Wavy',
      'Bell-shaped',
      'Umbrella-shaped',
    ];

    colors = [
      'White',
      'Beige',
      'Brown',
      'Red',
      'Orange',
      'Yellow',
      'Green',
      'Purple',
      'Black',
      'Gray',
    ];

    gillTypes = [
      'Free',
      'Attached',
      'Decurrent',
      'Subdecurrent',
      'Crowded',
      'Sparse',
      'Forked',
    ];

    stemTypes = [
      'Solid',
      'Hollow',
      'Fibrous',
      'Bulbous',
      'Rooted',
      'Ring/Annulus',
      'Cup/Volva',
    ];

    habitats = [
      'Forest',
      'Grassland',
      'Garden',
      'Dead wood',
      'Living tree',
      'Underground',
      'Disturbed soil',
    ];

    seasons = [
      'Spring',
      'Summer',
      'Autumn',
      'Winter',
    ];
  }

  /// Sets selected image metadata from camera or gallery.
  void setSelectedImage({
    required String imagePath,
    Uint8List? imageBytes,
  }) {
    try {
      selectedImagePath.value = imagePath;
      selectedImageBytes.value = imageBytes;
      _logger.i('Image selected: $imagePath');
    } catch (e) {
      _logger.e('Error setting selected image: $e');
      setError('Failed to select image');
    }
  }

  /// Clears the selected image and resets to camera step
  void clearImage() {
    selectedImagePath.value = null;
    selectedImageBytes.value = null;
    selectedTraits.clear();
    currentStep.value = 0;
    _logger.i('Image cleared, reset to camera step');
  }

  /// Updates a single trait selection
  void setTrait(String category, dynamic value) {
    try {
      selectedTraits[category] = value;
      _logger.i('Trait updated: $category = $value');
    } catch (e) {
      _logger.e('Error setting trait: $e');
      setError('Failed to update trait');
    }
  }

  /// Updates multiple traits at once
  void setTraits(Map<String, dynamic> traits) {
    try {
      selectedTraits.addAll(traits);
      _logger.i('Multiple traits updated: ${traits.keys.join(", ")}');
    } catch (e) {
      _logger.e('Error setting traits: $e');
      setError('Failed to update traits');
    }
  }

  /// Clears all selected traits
  void clearTraits() {
    selectedTraits.clear();
    _logger.i('All traits cleared');
  }

  /// Validates that required traits are selected
  /// 
  /// Required traits: cap_shape, color, gill_type, stem_type, habitat, season
  bool validateTraitsCompleted() {
    final requiredTraits = ['cap_shape', 'color', 'gill_type', 'stem_type', 'habitat', 'season'];
    final allSelected = requiredTraits.every((trait) => selectedTraits.containsKey(trait));

    if (!allSelected) {
      final missing = requiredTraits
          .where((t) => !selectedTraits.containsKey(t))
          .toList();
      setError('Missing traits: ${missing.join(", ")}');
      return false;
    }

    return true;
  }

  /// Advances to next step in identification flow
  void nextStep() {
    if (currentStep.value < 2) {
      currentStep.value++;
      _logger.i('Advanced to step ${currentStep.value}');
    }
  }

  /// Goes back to previous step
  void previousStep() {
    if (currentStep.value > 0) {
      currentStep.value--;
      _logger.i('Went back to step ${currentStep.value}');
    }
  }

  /// Navigates to specific step
  void goToStep(int step) {
    if (step >= 0 && step <= 2) {
      currentStep.value = step;
      _logger.i('Navigated to step $step');
    }
  }

  /// Sets processing flag (for showing loading indicators)
  void setProcessing(bool value) {
    isProcessing.value = value;
    _logger.i('Processing set to: $value');
  }

  /// Sets error message
  void setError(String? error) {
    errorMessage.value = error;
    if (error != null) {
      _logger.e('Error: $error');
    }
  }

  /// Clears error message
  void clearError() {
    errorMessage.value = null;
  }

  /// Gets all selected data for API submission
  /// 
  /// Returns a map containing:
  /// - imagePath: Path to image file
  /// - imageBytes: Raw image bytes when available (used on web)
  /// - traits: Selected traits dictionary
  Map<String, dynamic> getIdentificationData() {
    return {
      'imagePath': selectedImagePath.value,
      'imageBytes': selectedImageBytes.value,
      'traits': Map.from(selectedTraits),
    };
  }

  /// Resets all state to initial values
  void reset() {
    selectedImagePath.value = null;
    selectedImageBytes.value = null;
    selectedTraits.clear();
    isProcessing.value = false;
    errorMessage.value = null;
    currentStep.value = 0;
    _logger.i('Identification provider reset');
  }

  /// Gets a trait option by category
  List<String> getTraitOptions(String category) {
    switch (category.toLowerCase()) {
      case 'cap_shape':
      case 'capshape':
        return capShapes;
      case 'color':
        return colors;
      case 'gill_type':
      case 'gilltype':
        return gillTypes;
      case 'stem_type':
      case 'stemtype':
        return stemTypes;
      case 'habitat':
        return habitats;
      case 'season':
        return seasons;
      default:
        return [];
    }
  }

  /// Logs current state for debugging
  void logCurrentState() {
    _logger.i(
      'Current state: '
      'Step=${currentStep.value}, '
      'Image=${selectedImagePath.value}, '
      'Traits=${selectedTraits.length} selected, '
      'Processing=$isProcessing',
    );
  }
}
