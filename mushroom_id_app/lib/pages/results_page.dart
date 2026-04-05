import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../providers/history_provider.dart';
import '../providers/language_provider.dart';
import '../services/storage_service.dart';
import '../widgets/language_flag_button.dart';

/// Results page displaying mushroom identification results.
/// 
/// Shows:
/// - Overall confidence percentage (circular progress)
/// - Per-method confidence breakdown (Image/Trait/LLM)
/// - Top 5 predicted species with confidence scores
/// - Lookalike warnings (color-coded)
/// - Safety indicators
/// - Save/Share functionality
class ResultsPage extends StatefulWidget {
  const ResultsPage({Key? key}) : super(key: key);

  @override
  State<ResultsPage> createState() => _ResultsPageState();
}

class _ResultsPageState extends State<ResultsPage> {
  // Mock data - replace with actual API results
  late Map<String, dynamic> _results;
  bool _isDemoMode = false;
  String? _imagePath;
  Map<String, dynamic> _traits = {};
  String? _notes;
  late HistoryProvider _historyProvider;

  @override
  void initState() {
    super.initState();
    _historyProvider = Get.find<HistoryProvider>();
    final args = Get.arguments;
    final providedResults =
        args is Map<String, dynamic> ? args['results'] as Map<String, dynamic>? : null;
    _isDemoMode = args is Map<String, dynamic> && args['demoMode'] == true;
    _imagePath = args is Map<String, dynamic> ? args['imagePath'] as String? : null;
    _traits = args is Map<String, dynamic> && args['traits'] is Map
        ? Map<String, dynamic>.from(args['traits'] as Map)
        : {};
    _notes = args is Map<String, dynamic> ? args['notes'] as String? : null;

    _results = providedResults ??
        {
      'top_prediction': 'Boletus edulis',
      'overall_confidence': 0.87,
      'method_confidences': {
        'image': 0.92,
        'trait': 0.81,
        'llm': 0.88,
      },
      'predictions': [
        {'species': 'Boletus edulis', 'confidence': 0.87, 'common': 'Porcini', 'swedish_name': 'Karljohan'},
        {
          'species': 'Boletus reticulatus',
          'confidence': 0.78,
          'common': 'Summer Porcini',
          'swedish_name': 'Sommarsopp',
        },
        {'species': 'Xerocomelellus chrysenteron', 'confidence': 0.65, 'common': 'Red-Foot Bolete', 'swedish_name': 'Rödfotsopp'},
        {'species': 'Cep species', 'confidence': 0.62, 'common': 'King Bolete', 'swedish_name': 'Kungsopp'},
        {
          'species': 'Boletus luteus',
          'confidence': 0.55,
          'common': 'Slippery Jack',
          'swedish_name': 'Smörsopp',
        },
      ],
      'lookalikes': [
        {
          'species': 'Caloboletus calopus',
          'risk': 'high',
          'reason': 'Inedible, can cause stomach upset'
        },
        {
          'species': 'Boletus sensibilis',
          'risk': 'medium',
          'reason': 'Mild toxin, similar appearance'
        },
      ],
      'safety_rating': 'edible', // edible, caution, inedible, unknown
    };
  }

  /// Gets color based on confidence level
  Color _getConfidenceColor(double confidence) {
    if (confidence >= 0.8) return Colors.green;
    if (confidence >= 0.6) return Colors.orange;
    return Colors.red;
  }

  /// Gets color based on safety rating
  Color _getSafetyColor(String rating) {
    switch (rating) {
      case 'edible':
        return Colors.green;
      case 'caution':
        return Colors.orange;
      case 'inedible':
        return Colors.red;
      default:
        return Colors.blue;
    }
  }

  /// Gets emoji/icon for safety rating
  String _getSafetyIcon(String rating) {
    switch (rating) {
      case 'edible':
        return '✓';
      case 'caution':
        return '⚠';
      case 'inedible':
        return '✗';
      default:
        return '?';
    }
  }

  @override
  Widget build(BuildContext context) {
    final isMobile = MediaQuery.of(context).size.width < 600;
    final primaryColor = Theme.of(context).primaryColor;

    return Scaffold(
      appBar: AppBar(
        title: Text('identification_results'.tr),
        centerTitle: true,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.share),
            onPressed: _shareResults,
            tooltip: 'share_results'.tr,
          ),
          IconButton(
            icon: const Icon(Icons.bookmark),
            onPressed: _saveResults,
            tooltip: 'save_to_history'.tr,
          ),
          const LanguageFlagButton(),
        ],
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.all(isMobile ? 16.0 : 24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (_isDemoMode) ...[
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: Colors.orange.withOpacity(0.12),
                    border: Border.all(color: Colors.orange),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text('demo_notice'.tr),
                ),
                const SizedBox(height: 16),
              ],
              // Overall confidence section
              Obx(() => _buildConfidenceCard()),
              const SizedBox(height: 24),

              // Method breakdown section
              _buildMethodBreakdown(),
              const SizedBox(height: 24),

              // Safety indicator
              _buildSafetyIndicator(),
              const SizedBox(height: 24),

              // Top predictions
              Obx(() => _buildPredictionsSection()),
              const SizedBox(height: 24),

              // Lookalike warnings
              _buildLookalikesSection(),
              const SizedBox(height: 24),

              // Safety disclaimer
              _buildSafetyDisclaimer(),
              const SizedBox(height: 24),

              // Action buttons
              _buildActionButtons(),
            ],
          ),
        ),
      ),
    );
  }

  /// Builds main confidence display card
  Widget _buildConfidenceCard() {
    final overall = _results['overall_confidence'] as double;
    final topSpecies = _results['top_prediction'] as String;
    final confidencePercent = (overall * 100).toStringAsFixed(1);

    // Get common name for the top prediction
    final predictions = _results['top_predictions'] as List? ?? _results['predictions'] as List? ?? [];
    String commonName = '';
    if (predictions.isNotEmpty && predictions[0] is Map) {
      final top = predictions[0] as Map;
      final langProvider = Get.find<LanguageProvider>();
      final swedishName = top['swedish_name'] as String? ?? '';
      final englishName = top['common'] as String? ?? '';
      commonName = langProvider.isSwedish && swedishName.isNotEmpty ? swedishName : englishName;
    }

    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          children: [
            // Circular progress
            Text(
              '$confidencePercent%',
              style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: _getConfidenceColor(overall),
                  ),
            ),
            const SizedBox(height: 12),
            SizedBox(
              width: 150,
              height: 150,
              child: Stack(
                alignment: Alignment.center,
                children: [
                  // Background circle
                  Container(
                    width: 150,
                    height: 150,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: _getConfidenceColor(overall).withOpacity(0.1),
                    ),
                  ),
                  // Progress arc
                  CircularProgressIndicator(
                    value: overall,
                    strokeWidth: 12,
                    strokeCap: StrokeCap.round,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      _getConfidenceColor(overall),
                    ),
                  ),
                  // Center label
                  Text(
                    'confidence'.tr,
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // Top species name — local name first, Latin second
            if (commonName.isNotEmpty) ...[
              Text(
                commonName,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 4),
              Text(
                topSpecies,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      color: Colors.grey[600],
                      fontStyle: FontStyle.italic,
                    ),
                textAlign: TextAlign.center,
              ),
            ] else
              Text(
                topSpecies,
                style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
                textAlign: TextAlign.center,
              ),
            const SizedBox(height: 8),

            Text(
              'most_likely'.tr,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    color: Colors.grey[600],
                  ),
            ),
          ],
        ),
      ),
    );
  }

  /// Builds method confidence breakdown
  Widget _buildMethodBreakdown() {
    final methods = _results['method_confidences'] as Map<String, dynamic>;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'confidence_by_method'.tr,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 16),
        ...[
          ('image_recognition'.tr, methods['image'] ?? 0.0),
          ('trait_analysis'.tr, methods['trait'] ?? 0.0),
          ('language_model'.tr, methods['llm'] ?? 0.0),
        ].map((item) {
          final label = item.$1;
          final confidence = item.$2 as double;
          final percent = (confidence * 100).toStringAsFixed(0);

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(label),
                    Text(
                      '$percent%',
                      style: const TextStyle(fontWeight: FontWeight.bold),
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                ClipRRect(
                  borderRadius: BorderRadius.circular(4),
                  child: LinearProgressIndicator(
                    value: confidence,
                    minHeight: 8,
                    backgroundColor: Colors.grey[300],
                    valueColor: AlwaysStoppedAnimation<Color>(
                      _getConfidenceColor(confidence),
                    ),
                  ),
                ),
              ],
            ),
          );
        }).toList(),
      ],
    );
  }

  /// Builds safety indicator
  Widget _buildSafetyIndicator() {
    final safetyRating = _results['safety_rating'] as String;
    final color = _getSafetyColor(safetyRating);
    final icon = _getSafetyIcon(safetyRating);

    String safetyText = 'safety_unavailable'.tr;
    String safetyDescription = '';

    switch (safetyRating) {
      case 'edible':
        safetyText = 'likely_edible'.tr;
        safetyDescription = 'likely_edible_desc'.tr;
        break;
      case 'caution':
        safetyText = 'caution_advised'.tr;
        safetyDescription = 'caution_desc'.tr;
        break;
      case 'inedible':
        safetyText = 'not_recommended'.tr;
        safetyDescription = 'not_recommended_desc'.tr;
        break;
      case 'unknown':
        safetyText = 'safety_unknown'.tr;
        safetyDescription = 'safety_unknown_desc'.tr;
        break;
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        border: Border.all(color: color, width: 2),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: color.withOpacity(0.2),
            ),
            child: Center(
              child: Text(
                icon,
                style: TextStyle(
                  fontSize: 28,
                  color: color,
                ),
              ),
            ),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  safetyText,
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(
                        color: color,
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 4),
                Text(
                  safetyDescription,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[700],
                      ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  /// Builds top predictions list
  Widget _buildPredictionsSection() {
    final predictions = _results['predictions'] as List<dynamic>;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'top_predictions_label'.tr,
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
              ),
        ),
        const SizedBox(height: 12),
        ...predictions.asMap().entries.map((entry) {
          final index = entry.key;
          final prediction = entry.value as Map<String, dynamic>;
          final species = prediction['species'] as String;
          final confidence = prediction['confidence'] as double;
          final commonRaw = prediction['common'] as String? ?? '';
          final swedishRaw = prediction['swedish_name'] as String? ?? '';
          // Use swedish_name field directly when Swedish locale is active.
          final langProvider = Get.find<LanguageProvider>();
          final common = langProvider.isSwedish && swedishRaw.isNotEmpty
              ? swedishRaw
              : commonRaw;
          final percent = (confidence * 100).toStringAsFixed(0);

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              border: Border.all(color: Colors.grey[300]!),
              borderRadius: BorderRadius.circular(8),
              color: index == 0 ? Colors.green.withOpacity(0.05) : null,
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Text(
                                '#${index + 1}',
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  color: Colors.grey[600],
                                ),
                              ),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  common.isNotEmpty ? common : species,
                                  style: const TextStyle(
                                    fontWeight: FontWeight.bold,
                                    fontSize: 16,
                                  ),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 4),
                          Text(
                            species,
                            style: TextStyle(
                              color: Colors.grey[600],
                              fontSize: 13,
                              fontStyle: FontStyle.italic,
                            ),
                          ),
                        ],
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.end,
                      children: [
                        Text(
                          '$percent%',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: _getConfidenceColor(confidence),
                            fontSize: 16,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Container(
                          width: 60,
                          height: 4,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(2),
                            color: _getConfidenceColor(confidence).withOpacity(0.3),
                          ),
                          child: FractionallySizedBox(
                            widthFactor: confidence,
                            alignment: Alignment.centerLeft,
                            child: Container(
                              decoration: BoxDecoration(
                                borderRadius: BorderRadius.circular(2),
                                color: _getConfidenceColor(confidence),
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ],
            ),
          );
        }).toList(),
      ],
    );
  }

  /// Builds lookalike warnings section
  Widget _buildLookalikesSection() {
    final lookalikes = _results['lookalikes'] as List<dynamic>? ?? [];

    if (lookalikes.isEmpty) {
      return const SizedBox.shrink();
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '⚠️ ${'lookalike_warning'.tr}',
          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                fontWeight: FontWeight.bold,
                color: Colors.red[700],
              ),
        ),
        const SizedBox(height: 12),
        ...lookalikes.map((item) {
          final lookalike = item as Map<String, dynamic>;
          final species = lookalike['species'] as String;
          final risk = lookalike['risk'] as String;
          final reasonRaw = lookalike['reason'] as String;
          // Translate known demo reasons; fall back to raw value for real API.
          final reasonKey = species.contains('calopus')
              ? 'lookalike_reason_calopus'
              : species.contains('sensibilis')
                  ? 'lookalike_reason_sensibilis'
                  : null;
          final reason =
              reasonKey != null ? reasonKey.tr : reasonRaw;

          Color riskColor = Colors.red;
          if (risk == 'medium') riskColor = Colors.orange;

          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: riskColor.withOpacity(0.1),
              border: Border.all(color: riskColor),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Icon(
                      Icons.warning_rounded,
                      color: riskColor,
                      size: 20,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        species,
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                          color: riskColor,
                        ),
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: riskColor,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: Text(
                        risk.toUpperCase(),
                        style: const TextStyle(
                          fontSize: 11,
                          color: Colors.white,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                Text(
                  reason,
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[700],
                      ),
                ),
              ],
            ),
          );
        }).toList(),
      ],
    );
  }

  /// Builds safety disclaimer section
  Widget _buildSafetyDisclaimer() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.blue[50],
        border: Border.all(color: Colors.blue[200]!),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.info_outline, color: Colors.blue[700]),
              const SizedBox(width: 8),
              Text(
                'important_safety_notice'.tr,
                style: Theme.of(context).textTheme.labelLarge?.copyWith(
                      color: Colors.blue[900],
                      fontWeight: FontWeight.bold,
                    ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'safety_notice_text'.tr,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Colors.blue[900],
                ),
          ),
          const SizedBox(height: 8),
          ...[
            'safety_expert_mycologists'.tr,
            'safety_expert_poison'.tr,
            'safety_expert_guides'.tr,
          ]
              .map((item) => Padding(
                    padding: const EdgeInsets.only(bottom: 4, left: 16),
                    child: Text(
                      '• $item',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.blue[900],
                          ),
                    ),
                  ))
              .toList(),
        ],
      ),
    );
  }

  /// Builds action buttons at bottom
  Widget _buildActionButtons() {
    return Column(
      children: [
        Row(
          children: [
            Expanded(
              child: OutlinedButton(
                onPressed: () => Get.back(),
                child: Text('try_again'.tr),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: ElevatedButton(
                onPressed: _saveResults,
                child: Text('save_result'.tr),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        SizedBox(
          width: double.infinity,
          child: OutlinedButton.icon(
            onPressed: () => Get.offAllNamed('/'),
            icon: const Icon(Icons.home),
            label: Text('back_to_home'.tr),
          ),
        ),
      ],
    );
  }

  /// Shares results with other apps
  void _shareResults() {
    Get.snackbar(
      'share'.tr,
      'share_coming_soon'.tr,
      backgroundColor: Colors.blue[700],
      colorText: Colors.white,
    );
    // TODO: Implement share functionality
  }

  /// Saves result to history
  void _saveResults() {
    if (_imagePath == null || _imagePath!.isEmpty) {
      Get.snackbar(
        'save_failed'.tr,
        'no_image_path'.tr,
        backgroundColor: Colors.red[700],
        colorText: Colors.white,
      );
      return;
    }

    final historyResult = {
      ..._results,
      'top_predictions': _results['predictions'] ?? _results['top_predictions'] ?? [],
      'confidence': _results['overall_confidence'] ?? _results['confidence'] ?? 0.0,
    };

    _historyProvider
        .saveIdentification(
          HistoryEntry(
            imagePath: _imagePath!,
            traits: _traits,
            results: historyResult,
            createdAt: DateTime.now(),
            notes: (_notes == null || _notes!.isEmpty) ? null : _notes,
          ),
        )
        .then((_) {
          Get.snackbar(
            'saved'.tr,
            'result_saved'.tr,
            backgroundColor: Colors.green[700],
            colorText: Colors.white,
          );
        })
        .catchError((error) {
          Get.snackbar(
            'save_failed'.tr,
            error.toString(),
            backgroundColor: Colors.red[700],
            colorText: Colors.white,
          );
        });
  }
}
