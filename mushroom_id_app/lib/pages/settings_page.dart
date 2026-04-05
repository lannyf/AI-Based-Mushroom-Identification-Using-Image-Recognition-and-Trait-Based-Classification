import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:logger/logger.dart';
import '../services/storage_service.dart';
import '../widgets/language_flag_button.dart';

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
        SnackBar(content: Text('${'error_saving_setting'.tr}: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('settings'.tr),
        centerTitle: true,
        actions: const [LanguageFlagButton()],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // App Settings Section
            _SectionTitle('app_settings'.tr),
            Card(
              child: Column(
                children: [
                  _SwitchTile(
                    title: 'enable_notifications'.tr,
                    subtitle: 'notifications_subtitle'.tr,
                    value: _notificationsEnabled,
                    onChanged: (value) async {
                      setState(() => _notificationsEnabled = value);
                      await _saveSetting('notifications_enabled', value.toString());
                    },
                  ),
                  const Divider(height: 0),
                  _DropdownTile(
                    title: 'language'.tr,
                    subtitle: 'language_subtitle'.tr,
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
                    title: 'dark_mode'.tr,
                    subtitle: 'dark_mode_subtitle'.tr,
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
            _SectionTitle('api_settings'.tr),
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
                      label: Text('test_connection'.tr),
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
            _SectionTitle('storage_data'.tr),
            Card(
              child: Column(
                children: [
                  _SettingsTile(
                    title: 'clear_cache'.tr,
                    subtitle: 'clear_cache_subtitle'.tr,
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _showClearCacheDialog,
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'export_history'.tr,
                    subtitle: 'export_subtitle'.tr,
                    trailing: const Icon(Icons.chevron_right),
                    onTap: _exportHistory,
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'clear_all_data'.tr,
                    subtitle: 'clear_all_subtitle'.tr,
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
                  _SectionTitle('developer_tools'.tr),
                  Card(
                    child: Column(
                      children: [
                        _SettingsTile(
                          title: 'view_logs'.tr,
                          subtitle: 'view_logs_subtitle'.tr,
                          trailing: const Icon(Icons.chevron_right),
                          onTap: _showLogsDialog,
                        ),
                        const Divider(height: 0),
                        _SettingsTile(
                          title: 'database_info'.tr,
                          subtitle: 'database_info_subtitle'.tr,
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
            _SectionTitle('about'.tr),
            Card(
              child: Column(
                children: [
                  _SettingsTile(
                    title: 'app_version'.tr,
                    subtitle: '1.0.0',
                    trailing: Text('build_number'.tr),
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'privacy_policy'.tr,
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _showPrivacyPolicy(context),
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'terms_of_service'.tr,
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _showTermsOfService(context),
                  ),
                  const Divider(height: 0),
                  _SettingsTile(
                    title: 'about'.tr,
                    subtitle: 'about_subtitle'.tr,
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
                    _debugMode ? 'debug_mode_on'.tr : 'debug_mode_off'.tr,
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
        title: Text('connection_test_title'.tr),
        content: Text('connection_test_content'.tr),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('ok'.tr),
          ),
        ],
      ),
    );
  }

  void _showClearCacheDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('clear_cache_confirm_title'.tr),
        content: Text('clear_cache_confirm_text'.tr),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('cancel'.tr),
          ),
          TextButton(
            onPressed: () {
              // TODO: Implement cache clearing
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('cache_cleared'.tr)),
              );
            },
            child: Text('clear'.tr),
          ),
        ],
      ),
    );
  }

  void _showClearAllDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('clear_all_data_title'.tr),
        content: Text('clear_all_data_text'.tr),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('cancel'.tr),
          ),
          TextButton(
            onPressed: () {
              // TODO: Implement full data deletion
              Navigator.pop(context);
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(content: Text('all_data_deleted'.tr)),
              );
            },
            child: Text('delete_all'.tr, style: const TextStyle(color: Colors.red)),
          ),
        ],
      ),
    );
  }

  void _exportHistory() {
    // TODO: Implement history export
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text('export_coming_soon'.tr)),
    );
  }

  void _showLogsDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('application_logs_title'.tr),
        content: SingleChildScrollView(
          child: Text(
            'logs_placeholder'.tr,
            style: const TextStyle(fontFamily: 'Courier', fontSize: 11),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('close'.tr),
          ),
        ],
      ),
    );
  }

  void _showDatabaseInfo() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('database_information_title'.tr),
        content: SingleChildScrollView(
          child: Text('database_placeholder'.tr),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('close'.tr),
          ),
        ],
      ),
    );
  }

  void _showPrivacyPolicy(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('privacy_policy'.tr),
        content: SingleChildScrollView(
          child: Text('privacy_policy_text'.tr),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('close'.tr),
          ),
        ],
      ),
    );
  }

  void _showTermsOfService(BuildContext context) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('terms_of_service'.tr),
        content: SingleChildScrollView(
          child: Text('terms_text'.tr),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('close'.tr),
          ),
        ],
      ),
    );
  }

  void _showAbout() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text('about'.tr),
        content: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                'about_app_name'.tr,
                style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
              ),
              const SizedBox(height: 12),
              Text('about_app_desc'.tr),
              const SizedBox(height: 12),
              Text(
                'about_features'.tr,
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
              Text('about_features_list'.tr),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: Text('close'.tr),
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
