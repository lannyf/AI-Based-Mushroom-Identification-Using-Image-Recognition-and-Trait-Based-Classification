import 'package:flutter/material.dart';
import 'package:get/get.dart';
import 'package:shared_preferences/shared_preferences.dart';

class LanguageProvider extends GetxController {
  static const _prefKey = 'app_locale';

  final _locale = const Locale('en', 'US').obs;

  Locale get locale => _locale.value;
  bool get isSwedish => _locale.value.languageCode == 'sv';

  @override
  void onInit() {
    super.onInit();
    _loadLocale();
  }

  Future<void> _loadLocale() async {
    final prefs = await SharedPreferences.getInstance();
    final saved = prefs.getString(_prefKey);
    if (saved == 'sv') {
      _setLocale(const Locale('sv', 'SE'));
    }
  }

  void toggleLanguage() {
    if (isSwedish) {
      _setLocale(const Locale('en', 'US'));
    } else {
      _setLocale(const Locale('sv', 'SE'));
    }
  }

  void _setLocale(Locale locale) {
    _locale.value = locale;
    Get.updateLocale(locale);
    SharedPreferences.getInstance().then(
      (prefs) => prefs.setString(_prefKey, locale.languageCode),
    );
  }
}
