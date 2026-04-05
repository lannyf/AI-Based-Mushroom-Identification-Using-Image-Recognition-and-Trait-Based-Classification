import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'dart:typed_data';

import '../providers/identification_provider.dart';
import '../services/identification_api_service.dart';
import '../widgets/language_flag_button.dart';

/// Questionnaire page for trait selection.
/// 
/// Users select mushroom characteristics:
/// - Cap shape (convex, flat, conical, etc.)
/// - Color (white, brown, red, etc.)
/// - Gill type (free, attached, crowded, etc.)
/// - Stem type (solid, hollow, bulbous, etc.)
/// - Habitat (forest, garden, dead wood, etc.)
/// - Season (spring, summer, autumn, winter)
/// 
/// Plus optional notes field for additional observations.
class QuestionnairePage extends StatefulWidget {
  const QuestionnairePage({Key? key}) : super(key: key);

  @override
  State<QuestionnairePage> createState() => _QuestionnairePageState();
}

class _QuestionnairePageState extends State<QuestionnairePage> {
  late IdentificationProvider _provider;
  late TextEditingController _notesController;
  late PageController _pageController;
  final IdentificationApiService _apiService = IdentificationApiService();
  int _currentPage = 0;

  @override
  void initState() {
    super.initState();
    _provider = Get.find<IdentificationProvider>();
    _notesController = TextEditingController();
    _pageController = PageController();

    // Initialize selected image from navigation arguments, if provided.
    // Expected keys (via Get.toNamed): 'imagePath' (String), optional
    // 'imageName' (String) and 'imageBytes' (Uint8List).
    final args = Get.arguments;
    if (args is Map) {
      final String? imagePath = args['imagePath'] is String ? args['imagePath'] as String : null;
      final Uint8List? imageBytes =
          args['imageBytes'] is Uint8List ? args['imageBytes'] as Uint8List : null;

      if (imagePath != null && imagePath.isNotEmpty) {
        _provider.setSelectedImage(
          imagePath: imagePath,
          imageBytes: imageBytes,
        );
      }
    }
  }

  @override
  void dispose() {
    _notesController.dispose();
    _pageController.dispose();
    super.dispose();
  }

  /// Moves to next question page
  void _nextPage() {
    if (_currentPage < 5) {
      _pageController.nextPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  /// Moves to previous question page
  void _previousPage() {
    if (_currentPage > 0) {
      _pageController.previousPage(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
      );
    }
  }

  /// Validates all traits are selected and submits for identification
  void _submitQuestionnaire() {
    if (!_provider.validateTraitsCompleted()) {
      Get.snackbar(
        'missing_info'.tr,
        'missing_traits'.tr,
        backgroundColor: Colors.red[700],
        colorText: Colors.white,
      );
      return;
    }

    // Add notes if provided
    if (_notesController.text.isNotEmpty) {
      _provider.setTrait('notes', _notesController.text);
    }

    final identificationData = _provider.getIdentificationData();
    final String? imagePath = identificationData['imagePath'] as String?;
    final Uint8List? imageBytes = identificationData['imageBytes'] as Uint8List?;
    final Map<String, dynamic> traits =
        Map<String, dynamic>.from(identificationData['traits'] as Map);

    if (imagePath == null || imagePath.isEmpty) {
      Get.snackbar(
        'missing_image'.tr,
        'missing_image_desc'.tr,
        backgroundColor: Colors.red[700],
        colorText: Colors.white,
      );
      return;
    }

    _provider.setProcessing(true);

    _apiService
        .identifyMushroom(
          imagePath: imagePath,
          imageBytes: imageBytes,
          traits: traits,
        )
        .then((results) {
          _provider.setProcessing(false);
          Get.toNamed(
            '/results',
            arguments: {
              'demoMode': false,
              'results': results,
              'imagePath': imagePath,
              'traits': traits,
              'notes': _notesController.text.trim(),
            },
          );
        })
        .catchError((error) {
          _provider.setProcessing(false);
          Get.snackbar(
            'identification_failed'.tr,
            error.toString(),
            backgroundColor: Colors.red[700],
            colorText: Colors.white,
          );
        });
  }

  /// Handles trait selection from radio group
  void _selectTrait(String category, String value) {
    _provider.setTrait(category, value);
  }

  /// Returns the localised display label for a trait option value.
  /// The provider always stores the English value (used by the API).
  String _localizeTraitOption(String value) {
    final key = 'trait_$value';
    final translated = key.tr;
    // If GetX found no translation, it returns the key itself – fall back to value.
    return translated == key ? value : translated;
  }

  @override
  Widget build(BuildContext context) {
    final primaryColor = Theme.of(context).primaryColor;
    final isMobile = MediaQuery.of(context).size.width < 600;

    return WillPopScope(
      onWillPop: () async {
        if (_currentPage > 0) {
          _previousPage();
          return false;
        }
        return true;
      },
      child: Scaffold(
        appBar: AppBar(
          title: Text('mushroom_details'.tr),
          centerTitle: true,
          elevation: 0,
          leading: _currentPage == 0
              ? IconButton(
                  icon: const Icon(Icons.arrow_back),
                  onPressed: () => Get.back(),
                )
              : IconButton(
                  icon: const Icon(Icons.arrow_back),
                  onPressed: _previousPage,
                ),
          actions: const [LanguageFlagButton()],
        ),
        body: Column(
          children: [
            // Progress indicator
            LinearProgressIndicator(
              value: (_currentPage + 1) / 6,
              minHeight: 4,
              backgroundColor: Colors.grey[300],
              valueColor: AlwaysStoppedAnimation<Color>(primaryColor),
            ),

            // Question pages
            Expanded(
              child: PageView(
                controller: _pageController,
                onPageChanged: (page) {
                  setState(() {
                    _currentPage = page;
                  });
                },
                children: [
                  _buildCapShapePage(),
                  _buildColorPage(),
                  _buildGillTypePage(),
                  _buildStemTypePage(),
                  _buildHabitatPage(),
                  _buildSeasonPage(),
                ],
              ),
            ),

            // Bottom navigation and notes
            Container(
              color: Colors.grey[50],
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  if (_currentPage == 5) ...[
                    // Notes field (only on last page)
                    Text(
                      'notes_optional'.tr,
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _notesController,
                      maxLines: 3,
                      decoration: InputDecoration(
                        hintText: 'notes_hint'.tr,
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(8),
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),
                  ],

                  // Navigation buttons
                  Row(
                    children: [
                      // Previous button
                      if (_currentPage > 0)
                        Expanded(
                          child: OutlinedButton(
                            onPressed: _previousPage,
                            child: Text('previous'.tr),
                          ),
                        ),
                      if (_currentPage > 0) const SizedBox(width: 12),

                      // Next or Submit button
                      Expanded(
                        child: _currentPage < 5
                            ? ElevatedButton(
                                onPressed: _nextPage,
                                child: Text('next'.tr),
                              )
                            : ElevatedButton(
                                onPressed: _submitQuestionnaire,
                                child: Text('identify_mushroom'.tr),
                                style: ElevatedButton.styleFrom(
                                  backgroundColor: primaryColor,
                                  foregroundColor: Colors.white,
                                ),
                              ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  /// Builds cap shape question page
  Widget _buildCapShapePage() {
    final options = _provider.getTraitOptions('cap_shape');
    return _buildTraitPage(
      title: 'cap_shape_title'.tr,
      description: 'cap_shape_desc'.tr,
      imagePath: 'assets/images/cap-shapes.png', // TODO: Add image asset
      category: 'cap_shape',
      options: options,
    );
  }

  /// Builds color question page
  Widget _buildColorPage() {
    final options = _provider.getTraitOptions('color');
    return _buildTraitPage(
      title: 'cap_color_title'.tr,
      description: 'cap_color_desc'.tr,
      imagePath: 'assets/images/colors.png', // TODO: Add image asset
      category: 'color',
      options: options,
    );
  }

  /// Builds gill type question page
  Widget _buildGillTypePage() {
    final options = _provider.getTraitOptions('gill_type');
    return _buildTraitPage(
      title: 'gill_type_title'.tr,
      description: 'gill_type_desc'.tr,
      imagePath: 'assets/images/gill-types.png', // TODO: Add image asset
      category: 'gill_type',
      options: options,
    );
  }

  /// Builds stem type question page
  Widget _buildStemTypePage() {
    final options = _provider.getTraitOptions('stem_type');
    return _buildTraitPage(
      title: 'stem_type_title'.tr,
      description: 'stem_type_desc'.tr,
      imagePath: 'assets/images/stem-types.png', // TODO: Add image asset
      category: 'stem_type',
      options: options,
    );
  }

  /// Builds habitat question page
  Widget _buildHabitatPage() {
    final options = _provider.getTraitOptions('habitat');
    return _buildTraitPage(
      title: 'habitat_title'.tr,
      description: 'habitat_desc'.tr,
      imagePath: 'assets/images/habitats.png', // TODO: Add image asset
      category: 'habitat',
      options: options,
    );
  }

  /// Builds season question page
  Widget _buildSeasonPage() {
    final options = _provider.getTraitOptions('season');
    return _buildTraitPage(
      title: 'season_title'.tr,
      description: 'season_desc'.tr,
      imagePath: 'assets/images/seasons.png', // TODO: Add image asset
      category: 'season',
      options: options,
    );
  }

  /// Generic trait page builder
  Widget _buildTraitPage({
    required String title,
    required String description,
    required String imagePath,
    required String category,
    required List<String> options,
  }) {
    return Obx(() {
      final selectedValue = _provider.selectedTraits[category];

      return SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Question title
              Text(
                title,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 8),

              // Description
              Text(
                description,
                style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                      color: Colors.grey[600],
                    ),
              ),
              const SizedBox(height: 24),

              // Image placeholder (TODO: Add actual images)
              Container(
                width: double.infinity,
                height: 150,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: Center(
                  child: Icon(
                    Icons.image,
                    size: 48,
                    color: Colors.grey[400],
                  ),
                ),
              ),
              const SizedBox(height: 24),

              // Radio group options
              Text(
                'select_one'.tr,
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 12),

              ...options.map((option) {
                final isSelected = selectedValue == option;
                return Container(
                  margin: const EdgeInsets.only(bottom: 12),
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: isSelected
                          ? Theme.of(context).primaryColor
                          : Colors.grey[300]!,
                      width: isSelected ? 2 : 1,
                    ),
                    borderRadius: BorderRadius.circular(8),
                    color: isSelected
                        ? Theme.of(context).primaryColor.withOpacity(0.1)
                        : Colors.white,
                  ),
                  child: RadioListTile<String>(
                    title: Text(_localizeTraitOption(option)),
                    value: option,
                    groupValue: selectedValue,
                    onChanged: (value) {
                      if (value != null) {
                        _selectTrait(category, value);
                      }
                    },
                    activeColor: Theme.of(context).primaryColor,
                  ),
                );
              }).toList(),

              const SizedBox(height: 24),
            ],
          ),
        ),
      );
    });
  }
}
