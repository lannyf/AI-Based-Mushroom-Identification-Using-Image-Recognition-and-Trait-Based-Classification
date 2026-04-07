import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../providers/identification_provider.dart';
import '../providers/language_provider.dart';
import '../widgets/language_flag_button.dart';

/// Step 2 — Species Key Tree Traversal page.
///
/// Shows questions from the key.xml species decision tree one at a time.
/// The user taps an answer button; the engine continues until it reaches
/// a conclusion (species identified), at which point Steps 3 and 4 are
/// run automatically and the user is sent to ResultsPage.
class TreeTraversalPage extends StatefulWidget {
  const TreeTraversalPage({Key? key}) : super(key: key);

  @override
  State<TreeTraversalPage> createState() => _TreeTraversalPageState();
}

class _TreeTraversalPageState extends State<TreeTraversalPage> {
  final IdentificationProvider _provider = Get.find<IdentificationProvider>();

  bool _initialised = false;

  @override
  void initState() {
    super.initState();
    // Start traversal once after first frame
    WidgetsBinding.instance.addPostFrameCallback((_) => _startTraversal());
  }

  Future<void> _startTraversal() async {
    await _provider.runStep2Start();
    if (_provider.step2Concluded.value) {
      await _finalise();
    }
    setState(() => _initialised = true);
  }

  Future<void> _answer(String option) async {
    await _provider.runStep2Answer(option);
    if (_provider.step2Concluded.value) {
      await _finalise();
    }
  }

  Future<void> _finalise() async {
    // Run Steps 3 and 4 then navigate to results
    await _provider.runStep3AndStep4();
    if (_provider.errorMessage.value == null) {
      Get.toNamed('/results', arguments: {
        'results': _provider.step4Result.value,
        'imagePath': _provider.selectedImagePath.value,
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final isSwedish = Get.find<LanguageProvider>().isSwedish;
    final primaryColor = Theme.of(context).primaryColor;

    return Scaffold(
      appBar: AppBar(
        title: Text(isSwedish ? 'Artbestämning' : 'Species Identification'),
        actions: const [LanguageFlagButton()],
      ),
      body: Obx(() {
        if (_provider.isProcessing.value || !_initialised) {
          return _buildLoading(context, isSwedish);
        }

        if (_provider.errorMessage.value != null) {
          return _buildError(context, isSwedish, primaryColor);
        }

        if (_provider.step2Concluded.value) {
          return _buildConcluding(context, isSwedish);
        }

        return _buildQuestion(context, isSwedish, primaryColor);
      }),
    );
  }

  // ---------------------------------------------------------------------------
  // Loading
  // ---------------------------------------------------------------------------

  Widget _buildLoading(BuildContext context, bool isSwedish) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const CircularProgressIndicator(),
          const SizedBox(height: 24),
          Text(
            isSwedish ? 'Analyserar…' : 'Analysing…',
            style: Theme.of(context).textTheme.bodyLarge,
          ),
        ],
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Error
  // ---------------------------------------------------------------------------

  Widget _buildError(BuildContext context, bool isSwedish, Color primaryColor) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.error_outline, size: 64, color: Colors.red),
            const SizedBox(height: 16),
            Text(
              _provider.errorMessage.value!,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 24),
            ElevatedButton(
              onPressed: () => Get.back(),
              child: Text(isSwedish ? 'Tillbaka' : 'Back'),
            ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Concluding (steps 3+4 running)
  // ---------------------------------------------------------------------------

  Widget _buildConcluding(BuildContext context, bool isSwedish) {
    final species = _provider.step2Result.value?['species'] ?? '';
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const CircularProgressIndicator(),
            const SizedBox(height: 24),
            Text(
              isSwedish
                  ? 'Bekräftar: $species…'
                  : 'Confirming: $species…',
              style: Theme.of(context).textTheme.titleMedium,
              textAlign: TextAlign.center,
            ),
            const SizedBox(height: 8),
            Text(
              isSwedish ? 'Kontrollerar mot artdatabasen' : 'Verifying against species database',
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey[600]),
            ),
          ],
        ),
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Question + option buttons
  // ---------------------------------------------------------------------------

  Widget _buildQuestion(BuildContext context, bool isSwedish, Color primaryColor) {
    final question = _provider.step2Question.value ?? '';
    final options  = _provider.step2Options;
    final autoAnswers = _provider.step2AutoAnswers.value;
    final userAnswers = _provider.step2UserAnswers.value;

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Progress indicator
          _buildProgressBar(context, isSwedish, autoAnswers, userAnswers, primaryColor),
          const SizedBox(height: 32),

          // Auto-answered summary chip
          if (autoAnswers > 0) ...[
            Wrap(
              spacing: 8,
              children: [
                Chip(
                  avatar: const Icon(Icons.auto_awesome, size: 16),
                  label: Text(
                    isSwedish
                        ? '$autoAnswers svar från bilden'
                        : '$autoAnswers answered from image',
                    style: const TextStyle(fontSize: 12),
                  ),
                  backgroundColor: Colors.green[50],
                ),
              ],
            ),
            const SizedBox(height: 16),
          ],

          // Question text
          Text(
            question,
            style: Theme.of(context).textTheme.titleLarge?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
          ),
          const SizedBox(height: 8),
          Text(
            isSwedish
                ? 'Välj det alternativ som bäst stämmer med din svamp'
                : 'Select the option that best matches your mushroom',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey[600]),
          ),
          const SizedBox(height: 32),

          // Answer buttons
          ...options.map((option) => _buildOptionButton(context, option, primaryColor, isSwedish)),

          const SizedBox(height: 32),

          // Back button
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: () => Get.back(),
              icon: const Icon(Icons.arrow_back),
              label: Text(isSwedish ? 'Tillbaka' : 'Back'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressBar(
    BuildContext context,
    bool isSwedish,
    int autoAnswers,
    int userAnswers,
    Color primaryColor,
  ) {
    final total = autoAnswers + userAnswers;
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              isSwedish ? 'Artbestämning' : 'Species key',
              style: Theme.of(context).textTheme.labelLarge?.copyWith(color: Colors.grey[600]),
            ),
            Text(
              isSwedish ? '$total frågor besvarade' : '$total questions answered',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
            ),
          ],
        ),
        const SizedBox(height: 6),
        LinearProgressIndicator(
          value: total == 0 ? 0.05 : (total / (total + 4)).clamp(0.05, 0.9),
          backgroundColor: Colors.grey[200],
          valueColor: AlwaysStoppedAnimation<Color>(primaryColor),
          minHeight: 6,
          borderRadius: BorderRadius.circular(3),
        ),
      ],
    );
  }

  Widget _buildOptionButton(
    BuildContext context,
    String option,
    Color primaryColor,
    bool isSwedish,
  ) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: SizedBox(
        width: double.infinity,
        child: ElevatedButton(
          onPressed: () => _answer(option),
          style: ElevatedButton.styleFrom(
            alignment: Alignment.centerLeft,
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
            backgroundColor: Colors.white,
            foregroundColor: primaryColor,
            side: BorderSide(color: primaryColor.withOpacity(0.4)),
            elevation: 1,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          ),
          child: Row(
            children: [
              Icon(Icons.circle_outlined, size: 18, color: primaryColor.withOpacity(0.6)),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  option,
                  style: const TextStyle(fontSize: 15),
                ),
              ),
              Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey[400]),
            ],
          ),
        ),
      ),
    );
  }
}
