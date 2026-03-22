import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'dart:io';

import '../providers/identification_provider.dart';

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
  int _currentPage = 0;

  @override
  void initState() {
    super.initState();
    _provider = Get.find<IdentificationProvider>();
    _notesController = TextEditingController();
    _pageController = PageController();

    // Initialize selected image from navigation arguments, if provided.
    // Expected keys (via Get.toNamed): 'imageFile' (File) or 'imagePath' (String).
    final args = Get.arguments;
    if (args is Map) {
      final File? imageFile = args['imageFile'] is File ? args['imageFile'] as File : null;
      final String? imagePath = args['imagePath'] is String ? args['imagePath'] as String : null;

      if (imageFile != null) {
        _provider.setSelectedImage(imageFile);
      } else if (imagePath != null && imagePath.isNotEmpty) {
        _provider.setSelectedImage(File(imagePath));
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
        'Missing Information',
        'Please select all required traits',
        backgroundColor: Colors.red[700],
        colorText: Colors.white,
      );
      return;
    }

    // Add notes if provided
    if (_notesController.text.isNotEmpty) {
      _provider.setTrait('notes', _notesController.text);
    }

    // Show loading
    _provider.setProcessing(true);

    // TODO: Call API to identify mushroom
    Get.snackbar(
      'Success',
      'Submitting for identification...',
      backgroundColor: Colors.green[700],
      colorText: Colors.white,
    );

    // Simulate API call
    Future.delayed(const Duration(seconds: 2), () {
      _provider.setProcessing(false);
      // Navigate to results page once created
      Get.toNamed('/results');
    });
  }

  /// Handles trait selection from radio group
  void _selectTrait(String category, String value) {
    _provider.setTrait(category, value);
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
          title: const Text('Mushroom Details'),
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
                      'Additional Notes (Optional)',
                      style: Theme.of(context).textTheme.labelLarge?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 12),
                    TextField(
                      controller: _notesController,
                      maxLines: 3,
                      decoration: InputDecoration(
                        hintText: 'Any other observations about the mushroom?',
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
                            child: const Text('Previous'),
                          ),
                        ),
                      if (_currentPage > 0) const SizedBox(width: 12),

                      // Next or Submit button
                      Expanded(
                        child: _currentPage < 5
                            ? ElevatedButton(
                                onPressed: _nextPage,
                                child: const Text('Next'),
                              )
                            : ElevatedButton(
                                onPressed: _submitQuestionnaire,
                                child: const Text('Identify Mushroom'),
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
      title: 'Cap Shape',
      description: 'What shape is the mushroom cap?',
      imagePath: 'assets/images/cap-shapes.png', // TODO: Add image asset
      category: 'cap_shape',
      options: options,
    );
  }

  /// Builds color question page
  Widget _buildColorPage() {
    final options = _provider.getTraitOptions('color');
    return _buildTraitPage(
      title: 'Cap Color',
      description: 'What color is the mushroom cap?',
      imagePath: 'assets/images/colors.png', // TODO: Add image asset
      category: 'color',
      options: options,
    );
  }

  /// Builds gill type question page
  Widget _buildGillTypePage() {
    final options = _provider.getTraitOptions('gill_type');
    return _buildTraitPage(
      title: 'Gill Type',
      description: 'How are the gills attached to the stem?',
      imagePath: 'assets/images/gill-types.png', // TODO: Add image asset
      category: 'gill_type',
      options: options,
    );
  }

  /// Builds stem type question page
  Widget _buildStemTypePage() {
    final options = _provider.getTraitOptions('stem_type');
    return _buildTraitPage(
      title: 'Stem Type',
      description: 'What is the stem structure like?',
      imagePath: 'assets/images/stem-types.png', // TODO: Add image asset
      category: 'stem_type',
      options: options,
    );
  }

  /// Builds habitat question page
  Widget _buildHabitatPage() {
    final options = _provider.getTraitOptions('habitat');
    return _buildTraitPage(
      title: 'Habitat',
      description: 'Where did you find the mushroom?',
      imagePath: 'assets/images/habitats.png', // TODO: Add image asset
      category: 'habitat',
      options: options,
    );
  }

  /// Builds season question page
  Widget _buildSeasonPage() {
    final options = _provider.getTraitOptions('season');
    return _buildTraitPage(
      title: 'Season',
      description: 'When did you find the mushroom?',
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
                'Select one:',
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
                    title: Text(option),
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
