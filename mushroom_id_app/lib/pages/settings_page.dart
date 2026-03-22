import 'package:flutter/material.dart';
import 'package:logger/logger.dart';
import '../services/storage_service.dart';

class SettingsPage extends StatefulWidget {
  const SettingsPage({Key? key}) : super(key: key);

  @override
  State<SettingsPage> createState() => _SettingsPageState();
}

class _SettingsPageState extends State<SettingsPage> {
  final StorageService _storageService = StorageService();
  final _logger = Logger();

  // Settings state
  bool _notificationsEnabled = true;
  bool _darkModeEnabled = false;
  String _apiBaseUrl = 'http://localhost:8000';
  String _language = 'English';
  bool _debugMode = false;

  @override
  void initState() {
    super.initState();
    _loadSettings();
  }

  Future<void> _loadSettings() async {
    try {
      final notifs = await _storageService.getPreference('notifications_enabled');
      final darkMode = await _storageService.getPreference('dark_mode');
      final apiUrl = await _storageService.getPreference('api_base_url');
      final lang = await _storageService.getPreference('language');
      final debug = await _storageService.getPreference('debug_mode');

      setState(() {
        _notificationsEnabled = notifs == 'true' || notifs == null;
        _darkModeEnabled = darkMode == 'true';
        _apiBaseUrl = apiUrl ?? 'http://localhost:8000';
        _language = lang ?? 'English';
        _debugMode = debug == 'true';
      });
    } catch (e) {
      _logger.e('Error loading settings: $e');
    }
  }

  Future<void> _saveSetting(String key, String value) async {
    try {
      await _storageService.setPreference(key, value);
      _logger.i('Setting saved: $key = $value');
    } catch (e) {
      _logger.e('Error saving setting: $e');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error saving setting: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Settings'),
        centerTitle: true,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // App Settings Section
            _SectionTitle('App Settings'),
            Card(
              child: Column(
                children: [
                  _SwitchTile(
                    title: 'Enable Notifications',
                    subtitle: 'Get alerts for identification results',
                    value: _notificationsEnabled,
                    onChanged: (value) async {
                      setState(() => _notificationsEnabled = value);
                      await _saveSetting('notifications_enabled', value.toString());
                    },
                  ),
                  const Divider(height: 0),
                  _DropdownTile(
                    title: 'Language',
                    subtitle: 'Select app language',
                    value: _language,
                    options: const ['English', 'Swedish', 'German', 'French'],
                    onChanged: (value) async {
                      if (value != null) {
                        setState(() => _language = value);
                        await _saveSetting('language', value);
                      }
                    },
                  ),
                  const Divider(height: 0),
                  _SwitchTile(
                    title: 'Dark Mode',
                    subtitle: 'Use dark theme (beta)',
                    value: _darkModeEnabled,
                    onChanged: (value) async {
                      setState(() => _darkModeEnabled = value);
                      await _saveSetting('dark_mode', value.toString());
                      // TODO: Implement theme switching
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // API Settings Section
            _SectionTitle('API Configuration'),
            Card(
              child: Column(
                children: [
                  _ApiUrlTile(
                    title: 'Backend Server URL',
                    subtitle: 'API endpoint for identification',
                    value: _apiBaseUrl,
                    onChanged: (value) async {
                      setState(() => _apiBaseUrl = value);
                      await _saveSetting('api_base_url', value);
                    },
                  ),
                  const Divider(height: 0),
                  Padding(
                    padding: const EdgeInsets.all(16),
                    child: ElevatedButton.icon(
                      onPressed: _testConnection,
                      icon: const Icon(Icons.cloud_queue),
                      label: const Text('Test Connection'),
                      style: ElevatedButton.styleFrom(
                        minimumSize: const Size.fromHeight(48),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Storage Settings Section
            _SectionTitle('Storage & Data'),
            Card(
              child: Column(
                children: [
                  _SettingsTile(
                    title: 'Clear Cache',
                    subtitle: 'Remove cached images and data',
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _showClearCacheDialog,
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'Export History',
                    subtitle: 'Export identification history as JSON',
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _exportHistory,
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'Clear All Data',
                    subtitle: 'Permanently delete all saved identifications',
                    trailing: Icon(Icons.warning, color: Colors.red[700]),
                    onTap: _showClearAllDialog,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Developer Settings Section
            if (_debugMode)
              Column(
                children: [
                  _SectionTitle('Developer Tools'),
                  Card(
                    child: Column(
                      children: [
                        _SettingsTile(
                          title: 'View Logs',
                          subtitle: 'View application logs',
                          trailing: const Icon(Icons.chevron_right),
                          onTap: _showLogsDialog,
                        ),
                        const Divider(height: 0),
                        _SettingsTile(
                          title: 'Database Info',
                          subtitle: 'View database statistics',
                          trailing: const Icon(Icons.chevron_right),
                          onTap: _showDatabaseInfo,
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),
                ],
              ),

            // About Section
            _SectionTitle('About'),
            Card(
              child: Column(
                children: [
                  _SettingsTile(
                    title: 'App Version',
                    subtitle: '1.0.0',
                    trailing: const Text('Build 1'),
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'Privacy Policy',
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _showPrivacyPolicy(context),
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'Terms of Service',
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _showTermsOfService(context),
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'About',
                    subtitle: 'AI-based Mushroom Identification System',
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _showAbout,
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            // Debug toggle at bottom
            Center(
              child: GestureDetector(
                onTap: () {
                  setState(() => _debugMode = !_debugMode);
                  _saveSetting('debug_mode', _debugMode.toString());
                },
                child: Padding(
                  padding: const EdgeInsets.all(8),
                  child: Text(
                    'Debug Mode: ${_debugMode ? "ON" : "OFF"}',
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.grey[500],
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),
          ],
        ),
      ),
    );
  }

  void _testConnection() async {
    // TODO: Implement API health check
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Connection Test'),
        content: const Text('Testing connection to API server...'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _showClearCacheDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Clear Cache?'),
        content: const Text(
          'This will remove cached images and data. Your identification history will not be affected.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              // TODO: Implement cache clearing
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('Cache cleared')),
              );
            },
            child: const Text('Clear'),
          ),
        ],
      ),
    );
  }

  void _showClearAllDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete All Data?'),
        content: const Text(
          'This will permanently delete all saved identifications, history, and preferences. This action cannot be undone.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () {
              // TODO: Implement full data deletion
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(content: Text('All data deleted')),
              );
            },
            child: const Text('Delete All', style: TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _exportHistory() {
    // TODO: Implement history export
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(content: Text('Export functionality coming soon')),
    );
  }

  void _showLogsDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Application Logs'),
        content: const SingleChildScrollView(
          child: Text(
            'Logs would be displayed here.\nFeature in development.',
            style: TextStyle(fontFamily: 'Courier', fontSize: 11),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showDatabaseInfo() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Database Information'),
        content: const SingleChildScrollView(
          child: Text(
            'Database: mushroom_identification.db\nTables: history, preferences\nFeature in development.',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showPrivacyPolicy(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Privacy Policy'),
        content: const SingleChildScrollView(
          child: Text(
            'Your privacy is important to us.\n\n'
            'This app processes images locally and sends them to our API server for analysis.\n\n'
            'We do not store images longer than necessary for processing.\n\n'
            'Your identification history is stored locally on your device.',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showTermsOfService(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Terms of Service'),
        content: const SingleChildScrollView(
          child: Text(
            '1. Use License\n'
            'This app is provided "as is" without warranty.\n\n'
            '2. Identification Disclaimer\n'
            'Identifications provided by this app are for reference only. Always verify with experts before consuming any mushroom.\n\n'
            '3. Liability\n'
            'We are not liable for any consequences from using this app.',
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _showAbout() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('About'),
        content: const SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'Mushroom Identification System',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              SizedBox(height: 12),
              Text(
                'An AI-powered application that uses image recognition and machine learning to identify mushroom species.',
              ),
              SizedBox(height: 12),
              Text(
                'Features:',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              Text('• Image-based identification\n'
                  '• Trait-based classification\n'
                  '• LLM-powered analysis\n'
                  '• Safety warnings\n'
                  '• Identification history\n'
                  '• Offline support'),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  final String title;

  const _SectionTitle(this.title);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12, top: 8),
      child: Text(
        title,
        style: Theme.of(context).textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.bold,
              color: Colors.grey[700],
            ),
      ),
    );
  }
}

class _SwitchTile extends StatelessWidget {
  final String title;
  final String? subtitle;
  final bool value;
  final ValueChanged<bool> onChanged;

  const _SwitchTile({
    Key? key,
    required this.title,
    this.subtitle,
    required this.value,
    required this.onChanged,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(title),
      subtitle: subtitle != null ? Text(subtitle!) : null,
      trailing: Switch(
        value: value,
        onChanged: onChanged,
      ),
    );
  }
}

class _DropdownTile extends StatelessWidget {
  final String title;
  final String? subtitle;
  final String value;
  final List<String> options;
  final ValueChanged<String?> onChanged;

  const _DropdownTile({
    Key? key,
    required this.title,
    this.subtitle,
    required this.value,
    required this.options,
    required this.onChanged,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    title,
                    style: Theme.of(context).textTheme.titleSmall,
                  ),
                  if (subtitle != null)
                    Text(
                      subtitle!,
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Colors.grey[600],
                          ),
                    ),
                ],
              ),
              DropdownButton<String>(
                value: value,
                items: options.map((option) {
                  return DropdownMenuItem(
                    value: option,
                    child: Text(option),
                  );
                }).toList(),
                onChanged: onChanged,
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _ApiUrlTile extends StatefulWidget {
  final String title;
  final String? subtitle;
  final String value;
  final ValueChanged<String> onChanged;

  const _ApiUrlTile({
    Key? key,
    required this.title,
    this.subtitle,
    required this.value,
    required this.onChanged,
  }) : super(key: key);

  @override
  State<_ApiUrlTile> createState() => _ApiUrlTileState();
}

class _ApiUrlTileState extends State<_ApiUrlTile> {
  late TextEditingController _controller;

  @override
  void initState() {
    super.initState();
    _controller = TextEditingController(text: widget.value);
  }

  @override
  void didUpdateWidget(covariant _ApiUrlTile oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.value != widget.value) {
      _controller.text = widget.value;
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            widget.title,
            style: Theme.of(context).textTheme.titleSmall,
          ),
          if (widget.subtitle != null)
            Text(
              widget.subtitle!,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.grey[600],
                  ),
            ),
          const SizedBox(height: 12),
          TextField(
            controller: _controller,
            decoration: InputDecoration(
              hintText: 'http://localhost:8000',
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(8),
              ),
            ),
            onChanged: widget.onChanged,
          ),
        ],
      ),
    );
  }
}

class _SettingsTile extends StatelessWidget {
  final String title;
  final String? subtitle;
  final Widget? trailing;
  final VoidCallback? onTap;

  const _SettingsTile({
    Key? key,
    required this.title,
    this.subtitle,
    this.trailing,
    this.onTap,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(title),
      subtitle: subtitle != null ? Text(subtitle!) : null,
      trailing: trailing,
      onTap: onTap,
    );
  }
}
