import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';

void main() {
  group('IdentificationProvider', () {
    late IdentificationProvider provider;

    setUp(() {
      // Register provider for testing
      if (Get.isRegistered<IdentificationProvider>()) {
        Get.delete<IdentificationProvider>();
      }
      provider = Get.put(IdentificationProvider());
    });

    tearDown(() {
      Get.delete<IdentificationProvider>();
    });

    group('Initialization', () {
      test('Provider initializes with correct trait options', () {
        expect(
          provider.capShapes.isNotEmpty,
          isTrue,
          reason: 'Cap shapes should be initialized',
        );
        expect(
          provider.colors.isNotEmpty,
          isTrue,
          reason: 'Colors should be initialized',
        );
        expect(
          provider.gillTypes.isNotEmpty,
          isTrue,
          reason: 'Gill types should be initialized',
        );
        expect(
          provider.stemTypes.isNotEmpty,
          isTrue,
          reason: 'Stem types should be initialized',
        );
        expect(
          provider.habitats.isNotEmpty,
          isTrue,
          reason: 'Habitats should be initialized',
        );
        expect(
          provider.seasons.isNotEmpty,
          isTrue,
          reason: 'Seasons should be initialized',
        );
      });

      test('Provider initializes with default values', () {
        expect(provider.currentStep.value, equals(0));
        expect(provider.isProcessing.value, isFalse);
        expect(provider.selectedTraits.isEmpty, isTrue);
      });
    });

    group('Trait management', () {
      test('setTrait adds trait to selectedTraits', () {
        provider.setTrait('cap_shape', 'convex');
        expect(
          provider.selectedTraits['cap_shape'],
          equals('convex'),
          reason: 'Should set cap_shape trait',
        );
      });

      test('setTrait updates existing trait', () {
        provider.setTrait('cap_shape', 'convex');
        provider.setTrait('cap_shape', 'flat');
        expect(
          provider.selectedTraits['cap_shape'],
          equals('flat'),
          reason: 'Should update cap_shape trait',
        );
      });

      test('clearTraits removes all traits', () {
        provider.setTrait('cap_shape', 'convex');
        provider.setTrait('color', 'brown');
        provider.clearTraits();
        expect(
          provider.selectedTraits.isEmpty,
          isTrue,
          reason: 'Should clear all traits',
        );
      });

      test('setTraits adds multiple traits', () {
        final traits = {
          'cap_shape': 'convex',
          'color': 'brown',
          'gill_type': 'free',
        };
        provider.setTraits(traits);
        expect(
          provider.selectedTraits['cap_shape'],
          equals('convex'),
        );
        expect(
          provider.selectedTraits['color'],
          equals('brown'),
        );
        expect(
          provider.selectedTraits['gill_type'],
          equals('free'),
        );
      });
    });

    group('Validation', () {
      test('validateTraitsCompleted returns false with no traits', () {
        final result = provider.validateTraitsCompleted();
        expect(
          result,
          isFalse,
          reason: 'Should fail validation with no traits',
        );
      });

      test('validateTraitsCompleted returns false with incomplete traits', () {
        provider.setTrait('cap_shape', 'convex');
        provider.setTrait('color', 'brown');
        // Missing other traits
        final result = provider.validateTraitsCompleted();
        expect(
          result,
          isFalse,
          reason: 'Should fail validation with incomplete traits',
        );
      });

      test('validateTraitsCompleted returns true with all traits', () {
        provider.setTrait('cap_shape', 'convex');
        provider.setTrait('color', 'brown');
        provider.setTrait('gill_type', 'free');
        provider.setTrait('stem_type', 'equal');
        provider.setTrait('habitat', 'forest');
        provider.setTrait('season', 'spring');

        final result = provider.validateTraitsCompleted();
        expect(
          result,
          isTrue,
          reason: 'Should pass validation with all traits',
        );
      });
    });

    group('Navigation', () {
      test('nextStep increments current step', () {
        expect(provider.currentStep.value, equals(0));
        provider.nextStep();
        expect(provider.currentStep.value, equals(1));
        provider.nextStep();
        expect(provider.currentStep.value, equals(2));
      });

      test('previousStep decrements current step', () {
        provider.currentStep.value = 2;
        provider.previousStep();
        expect(provider.currentStep.value, equals(1));
        provider.previousStep();
        expect(provider.currentStep.value, equals(0));
      });

      test('previousStep does not go below 0', () {
        provider.currentStep.value = 0;
        provider.previousStep();
        expect(
          provider.currentStep.value,
          equals(0),
          reason: 'Should not go below 0',
        );
      });

      test('goToStep sets current step to specified value', () {
        provider.goToStep(2);
        expect(provider.currentStep.value, equals(2));
        provider.goToStep(0);
        expect(provider.currentStep.value, equals(0));
      });
    });

    group('Data collection', () {
      test('getIdentificationData returns complete object', () {
        // Set up test data
        provider.setTrait('cap_shape', 'convex');
        provider.setTrait('color', 'brown');
        provider.setTrait('gill_type', 'free');
        provider.setTrait('stem_type', 'equal');
        provider.setTrait('habitat', 'forest');
        provider.setTrait('season', 'spring');

        final data = provider.getIdentificationData();

        expect(
          data.containsKey('traits'),
          isTrue,
          reason: 'Should include traits',
        );
        expect(
          data['traits'],
          isNotNull,
          reason: 'Traits should not be null',
        );
      });

      test('getIdentificationData includes all selected traits', () {
        final expectedTraits = {
          'cap_shape': 'convex',
          'color': 'brown',
          'gill_type': 'free',
        };

        provider.setTraits(expectedTraits);
        final data = provider.getIdentificationData();
        final returnedTraits = data['traits'] as Map<String, dynamic>;

        expect(
          returnedTraits['cap_shape'],
          equals('convex'),
        );
        expect(
          returnedTraits['color'],
          equals('brown'),
        );
        expect(
          returnedTraits['gill_type'],
          equals('free'),
        );
      });
    });

    group('Trait options retrieval', () {
      test('getTraitOptions returns correct list for cap_shape', () {
        final options = provider.getTraitOptions('cap_shape');
        expect(
          options.isNotEmpty,
          isTrue,
          reason: 'Should return cap shape options',
        );
        expect(
          options.contains('Convex'),
          isTrue,
          reason: 'Should include convex option',
        );
      });

      test('getTraitOptions returns correct list for color', () {
        final options = provider.getTraitOptions('color');
        expect(
          options.isNotEmpty,
          isTrue,
          reason: 'Should return color options',
        );
      });

      test('getTraitOptions returns empty for unknown category', () {
        final options = provider.getTraitOptions('unknown_category');
        expect(
          options.isEmpty,
          isTrue,
          reason: 'Should return empty list for unknown category',
        );
      });

      test('All trait categories have options', () {
        expect(provider.getTraitOptions('cap_shape').length, equals(6));
        expect(provider.getTraitOptions('color').length, equals(10));
        expect(provider.getTraitOptions('gill_type').length, equals(7));
        expect(provider.getTraitOptions('stem_type').length, equals(7));
        expect(provider.getTraitOptions('habitat').length, equals(7));
        expect(provider.getTraitOptions('season').length, equals(4));
      });
    });

    group('State reset', () {
      test('reset clears all state', () {
        provider.setTrait('cap_shape', 'convex');
        provider.currentStep.value = 2;
        provider.errorMessage.value = 'Test error';

        provider.reset();

        expect(
          provider.selectedTraits.isEmpty,
          isTrue,
          reason: 'Traits should be cleared',
        );
        expect(
          provider.currentStep.value,
          equals(0),
          reason: 'Step should be reset to 0',
        );
        expect(
          provider.errorMessage.value,
          isEmpty,
          reason: 'Error message should be cleared',
        );
      });
    });
  });
}
