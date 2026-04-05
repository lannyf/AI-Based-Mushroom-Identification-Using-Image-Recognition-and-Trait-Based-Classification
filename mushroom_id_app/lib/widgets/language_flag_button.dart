import 'package:flutter/material.dart';
import 'package:get/get.dart';

import '../providers/language_provider.dart';

/// A flag button placed in AppBar actions that toggles between
/// Swedish 🇸🇪 and English 🇬🇧 locales.
class LanguageFlagButton extends StatelessWidget {
  const LanguageFlagButton({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final controller = Get.find<LanguageProvider>();

    return Obx(
      () => Tooltip(
        message: controller.isSwedish ? 'Switch to English' : 'Byt till svenska',
        child: InkWell(
          borderRadius: BorderRadius.circular(20),
          onTap: controller.toggleLanguage,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            child: Text(
              controller.isSwedish ? '🇸🇪' : '🇬🇧',
              style: const TextStyle(fontSize: 24),
            ),
          ),
        ),
      ),
    );
  }
}
