import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:intl/intl.dart';
import '../providers/history_provider.dart';
import '../services/storage_service.dart';

class HistoryPage extends StatefulWidget {
  const HistoryPage({Key? key}) : super(key: key);

  @override
  State<HistoryPage> createState() => _HistoryPageState();
}

class _HistoryPageState extends State<HistoryPage> {
  late HistoryProvider _historyProvider;

  @override
  void initState() {
    super.initState();
    _historyProvider = Get.find<HistoryProvider>();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Identification History'),
        centerTitle: true,
        elevation: 0,
      ),
      body: Obx(() {
        if (_historyProvider.isLoading.value) {
          return const Center(
            child: CircularProgressIndicator(),
          );
        }

        if (_historyProvider.isEmpty) {
          return Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.history,
                  size: 64,
                  color: Colors.grey[400],
                ),
                const SizedBox(height: 16),
                Text(
                  'No identifications yet',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        color: Colors.grey[600],
                      ),
                ),
                const SizedBox(height: 8),
                Text(
                  'Start by taking a photo of a mushroom',
                  style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                        color: Colors.grey[500],
                      ),
                ),
              ],
            ),
          );
        }

        return ListView.builder(
          itemCount: _historyProvider.history.length,
          padding: const EdgeInsets.all(8),
          itemBuilder: (context, index) {
            final entry = _historyProvider.history[index];
            return _HistoryCard(
              entry: entry,
              onTap: () => _showDetailView(entry),
              onDelete: () => _deleteEntry(entry),
            );
          },
        );
      }),
      floatingActionButton: Obx(() {
        if (_historyProvider.isEmpty) return const SizedBox.shrink();
        return FloatingActionButton(
          onPressed: _showClearConfirmation,
          tooltip: 'Clear history',
          child: const Icon(Icons.delete_sweep),
        );
      }),
      floatingActionButtonLocation: FloatingActionButtonLocation.endFloat,
    );
  }

  void _showDetailView(HistoryEntry entry) {
    Get.to(
      () => HistoryDetailPage(entry: entry),
      transition: Transition.rightToLeft,
    );
  }

  void _deleteEntry(HistoryEntry entry) {
    Get.dialog(
      AlertDialog(
        title: const Text('Delete Entry?'),
        content: Text(
          'Are you sure you want to delete this identification of ${entry.topSpecies}?',
        ),
        actions: [
          TextButton(
            onPressed: () => Get.back(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              try {
                await _historyProvider.deleteEntry(entry.id!);
                if (!mounted) return;
                Get.back();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Identification deleted')),
                );
              } catch (e) {
                if (!mounted) return;
                Get.back();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Failed to delete identification')),
                );
              }
            },
            child: const Text('Delete', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _showClearConfirmation() {
    Get.dialog(
      AlertDialog(
        title: const Text('Clear History?'),
        content: const Text(
          'This will permanently delete all saved identifications. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Get.back(),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () async {
              try {
                await _historyProvider.clearAllHistory();
                if (!mounted) return;
                Get.back();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('History cleared')),
                );
              } catch (e) {
                if (!mounted) return;
                Get.back();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Failed to clear history')),
                );
              }
            },
            child: const Text('Clear', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }
}

class _HistoryCard extends StatelessWidget {
  final HistoryEntry entry;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const _HistoryCard({
    Key? key,
    required this.entry,
    required this.onTap,
    required this.onDelete,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final confidence = entry.confidence;
    final species = entry.topSpecies;
    final safetyRating = entry.safetyRating ?? 'Unknown';
    final dateFormat = DateFormat('MMM d, yyyy • HH:mm');
    final formattedDate = dateFormat.format(entry.createdAt);

    Color getSafetyColor() {
      switch (safetyRating.toLowerCase()) {
        case 'edible':
          return Colors.green;
        case 'caution':
          return Colors.orange;
        case 'inedible':
          return Colors.red;
        default:
          return Colors.grey;
      }
    }

    return Card(
      margin: const EdgeInsets.symmetric(vertical: 8, horizontal: 0),
      child: InkWell(
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              // Top row: Species and date
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          species,
                          style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                fontWeight: FontWeight.bold,
                              ),
                          maxLines: 1,
                          overflow: TextOverflow.ellipsis,
                        ),
                        const SizedBox(height: 4),
                        Text(
                          formattedDate,
                          style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                color: Colors.grey[600],
                              ),
                        ),
                      ],
                    ),
                  ),
                  // Confidence badge
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: confidence >= 0.80
                          ? Colors.green[50]
                          : confidence >= 0.60
                              ? Colors.orange[50]
                              : Colors.red[50],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      '${(confidence * 100).toStringAsFixed(0)}%',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                        color: confidence >= 0.80
                            ? Colors.green[700]
                            : confidence >= 0.60
                                ? Colors.orange[700]
                                : Colors.red[700],
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 12),

              // Safety rating and confidence bar
              Row(
                children: [
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: confidence,
                        minHeight: 6,
                        backgroundColor: Colors.grey[300],
                        valueColor: AlwaysStoppedAnimation<Color>(
                          confidence >= 0.80
                              ? Colors.green
                              : confidence >= 0.60
                                  ? Colors.orange
                                  : Colors.red,
                        ),
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: getSafetyColor().withOpacity(0.2),
                      borderRadius: BorderRadius.circular(6),
                    ),
                    child: Text(
                      safetyRating,
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: getSafetyColor(),
                      ),
                    ),
                  ),
                ],
              ),

              // Notes if available
              if (entry.notes != null && entry.notes!.isNotEmpty) ...[
                const SizedBox(height: 12),
                Text(
                  'Notes: ${entry.notes}',
                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                        color: Colors.grey[700],
                        fontStyle: FontStyle.italic,
                      ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],

              // Action buttons
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  TextButton.icon(
                    onPressed: onDelete,
                    icon: const Icon(Icons.delete_outline, size: 18),
                    label: const Text('Delete'),
                    style: TextButton.styleFrom(
                      foregroundColor: Colors.red,
                    ),
                  ),
                  const SizedBox(width: 8),
                  TextButton.icon(
                    onPressed: onTap,
                    icon: const Icon(Icons.arrow_forward, size: 18),
                    label: const Text('View'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class HistoryDetailPage extends StatelessWidget {
  final HistoryEntry entry;

  const HistoryDetailPage({Key? key, required this.entry}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final confidence = entry.confidence;
    final species = entry.topSpecies;
    final safetyRating = entry.safetyRating ?? 'Unknown';
    final dateFormat = DateFormat('MMMM d, yyyy • HH:mm:ss');
    final formattedDate = dateFormat.format(entry.createdAt);

    return Scaffold(
      appBar: AppBar(
        title: const Text('Identification Details'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Main result card
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Species Identified',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.grey[600],
                            fontWeight: FontWeight.w600,
                          ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      species,
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                    ),
                    const SizedBox(height: 16),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Confidence',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: Colors.grey[600],
                                  ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              '${(confidence * 100).toStringAsFixed(1)}%',
                              style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                    fontWeight: FontWeight.bold,
                                  ),
                            ),
                          ],
                        ),
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Safety',
                              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                    color: Colors.grey[600],
                                  ),
                            ),
                            const SizedBox(height: 4),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 12,
                                vertical: 6,
                              ),
                              decoration: BoxDecoration(
                                color: _getSafetyColor(safetyRating).withOpacity(0.2),
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                safetyRating,
                                style: TextStyle(
                                  fontWeight: FontWeight.bold,
                                  color: _getSafetyColor(safetyRating),
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Traits used
            Text(
              'Traits Selected',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            _buildTraitsWidget(context),
            const SizedBox(height: 24),

            // Timestamp
            Text(
              'Identification Details',
              style: Theme.of(context).textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.bold,
                  ),
            ),
            const SizedBox(height: 12),
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    _detailRow(context, 'Date & Time', formattedDate),
                    const Divider(),
                    _detailRow(
                      context,
                      'Image',
                      entry.imagePath.split('/').last,
                    ),
                    if (entry.notes != null && entry.notes!.isNotEmpty) ...[
                      const Divider(),
                      _detailRow(context, 'Notes', entry.notes!),
                    ],
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),

            // Full results JSON (for debugging)
            if (entry.results.isNotEmpty)
              Expandable(
                title: 'Full Results Data',
                child: SingleChildScrollView(
                  scrollDirection: Axis.horizontal,
                  child: Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey[100],
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: Text(
                      _formatJson(entry.results),
                      style: const TextStyle(
                        fontSize: 11,
                        fontFamily: 'Courier',
                      ),
                    ),
                  ),
                ),
              ),
          ],
        ),
      ),
      floatingActionButton: FloatingActionButton.extended(
        onPressed: () => _shareIdentification(context),
        icon: const Icon(Icons.share),
        label: const Text('Share'),
      ),
    );
  }

  Widget _buildTraitsWidget(BuildContext context) {
    final traits = entry.traits;
    if (traits.isEmpty) {
      return Card(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Text(
            'No traits recorded',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Colors.grey[600],
                ),
          ),
        ),
      );
    }

    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: traits.entries.map((entry) {
        return Chip(
          label: Text('${entry.key}: ${entry.value}'),
          backgroundColor: Colors.blue[50],
          labelStyle: TextStyle(color: Colors.blue[900]),
        );
      }).toList(),
    );
  }

  Widget _detailRow(BuildContext context, String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            label,
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  color: Colors.grey[600],
                  fontWeight: FontWeight.w600,
                ),
          ),
          const SizedBox(height: 4),
          Text(
            value,
            style: Theme.of(context).textTheme.bodyMedium,
          ),
        ],
      ),
    );
  }

  String _formatJson(Map<String, dynamic> json) {
    try {
      return const JsonEncoder.withIndent('  ').convert(json);
    } on JsonUnsupportedObjectError {
      // Fallback when the map contains non-JSON-serializable values (e.g. Uint8List, custom objects)
      return json.toString();
    }
  }

  Color _getSafetyColor(String safety) {
    switch (safety.toLowerCase()) {
      case 'edible':
        return Colors.green;
      case 'caution':
        return Colors.orange;
      case 'inedible':
        return Colors.red;
      default:
        return Colors.grey;
    }
  }

  void _shareIdentification(BuildContext context) {
    final text = '''
Mushroom Identification Results
Species: ${entry.topSpecies}
Confidence: ${(entry.confidence * 100).toStringAsFixed(1)}%
Safety: ${entry.safetyRating}
Date: ${entry.createdAt}
''';

    // TODO: Integrate with share_plus package
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Share functionality coming soon')),
    );
  }
}

class Expandable extends StatefulWidget {
  final String title;
  final Widget child;

  const Expandable({
    Key? key,
    required this.title,
    required this.child,
  }) : super(key: key);

  @override
  State<Expandable> createState() => _ExpandableState();
}

class _ExpandableState extends State<Expandable> {
  bool _isExpanded = false;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        InkWell(
          onTap: () => setState(() => _isExpanded = !_isExpanded),
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  widget.title,
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                      ),
                ),
                Icon(
                  _isExpanded ? Icons.expand_less : Icons.expand_more,
                  color: Colors.grey[600],
                ),
              ],
            ),
          ),
        ),
        if (_isExpanded) widget.child,
      ],
    );
  }
}
