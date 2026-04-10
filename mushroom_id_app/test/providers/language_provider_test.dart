import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/providers/language_provider.dart';
import 'package:shared_preferences/shared_preferences.dart';

void main() {
  group('LanguageProvider', () {
    late LanguageProvider provider;

    setUp(() async {
      SharedPreferences.setMockInitialValues({});
      Get.reset();
      provider = Get.put(LanguageProvider());
      // Allow onInit async work to settle
      await Future.delayed(Duration.zero);
    });

    tearDown(() {
      Get.reset();
    });

    group('Initial state', () {
      test('Default locale is English', () {
        expect(provider.locale.languageCode, equals('en'));
      });

      test('isSwedish is false by default', () {
        expect(provider.isSwedish, isFalse);
      });
    });

    group('toggleLanguage', () {
      test('toggleLanguage switches to Swedish', () {
        expect(provider.isSwedish, isFalse);
        provider.toggleLanguage();
        expect(provider.isSwedish, isTrue);
      });

      test('toggleLanguage switches back to English', () {
        provider.toggleLanguage(); // → Swedish
        provider.toggleLanguage(); // → English
        expect(provider.isSwedish, isFalse);
      });

      test('toggleLanguage changes locale language code', () {
        provider.toggleLanguage();
        expect(provider.locale.languageCode, equals('sv'));
      });

      test('Swedish locale has correct country code', () {
        provider.toggleLanguage();
        expect(provider.locale.countryCode, equals('SE'));
      });

      test('English locale has correct country code', () {
        expect(provider.locale.countryCode, equals('US'));
      });
    });

    group('Reactive state', () {
      test('locale is observable (Rx)', () {
        // Rx value should update when toggled
        final initialLocale = provider.locale;
        provider.toggleLanguage();
        expect(provider.locale, isNot(equals(initialLocale)));
      });

      test('Multiple toggles are consistent', () {
        for (int i = 0; i < 6; i++) {
          provider.toggleLanguage();
        }
        // 6 toggles → back to English
        expect(provider.isSwedish, isFalse);
      });
    });
  });
}
