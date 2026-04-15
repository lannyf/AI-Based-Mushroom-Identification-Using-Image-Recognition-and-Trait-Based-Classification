import 'package:flutter_test/flutter_test.dart';
import 'package:get/get.dart';
import 'package:mushroom_identification/providers/identification_provider.dart';
import 'package:mushroom_identification/services/identification_api_service.dart';

// ---------------------------------------------------------------------------
// Fake API service for pipeline tests
// ---------------------------------------------------------------------------

class _FakeApiService extends IdentificationApiService {
  Map<String, dynamic>? step2Response;
  Map<String, dynamic>? step3Response;
  Map<String, dynamic>? step4Response;
  Exception? errorToThrow;

  @override
  Future<Map<String, dynamic>> step2Start({
    required Map<String, dynamic> visibleTraits,
    String? sessionId,
  }) async {
    if (errorToThrow != null) throw errorToThrow!;
    return step2Response!;
  }

  @override
  Future<Map<String, dynamic>> step2Answer({
    required String sessionId,
    required String answer,
  }) async {
    if (errorToThrow != null) throw errorToThrow!;
    return step2Response!;
  }

  @override
  Future<Map<String, dynamic>> step3Compare({
    required String swedishName,
    required Map<String, dynamic> visibleTraits,
  }) async {
    if (errorToThrow != null) throw errorToThrow!;
    return step3Response!;
  }

  @override
  Future<Map<String, dynamic>> step4Finalize({
    required Map<String, dynamic> step1Result,
    required Map<String, dynamic> step2Result,
    required Map<String, dynamic> step3Result,
  }) async {
    if (errorToThrow != null) throw errorToThrow!;
    return step4Response!;
  }
}

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

  // -------------------------------------------------------------------------
  // Pipeline tests — require fake API service via constructor injection
  // -------------------------------------------------------------------------
  group('IdentificationProvider pipeline', () {
    late _FakeApiService fakeApi;
    late IdentificationProvider provider;

    setUp(() {
      fakeApi = _FakeApiService();
      if (Get.isRegistered<IdentificationProvider>()) {
        Get.delete<IdentificationProvider>();
      }
      provider = Get.put(IdentificationProvider(api: fakeApi));
    });

    tearDown(() {
      Get.delete<IdentificationProvider>();
    });

    group('_applyStep2Response — question shape', () {
      test('sets question, options, and clears concluded when status is question', () async {
        provider.step1Result.value = {
          'trait_extraction': {'visible_traits': <String, dynamic>{}},
        };
        fakeApi.step2Response = {
          'status': 'question',
          'session_id': 'sess-1',
          'question': 'What colour is the cap?',
          'options': ['Red', 'Brown', 'White'],
          'auto_answered': ['q0', 'q1'],
          'path': ['q0', 'q1'],
        };

        await provider.runStep2Start();

        expect(provider.step2SessionId.value, equals('sess-1'));
        expect(provider.step2Question.value, equals('What colour is the cap?'));
        expect(provider.step2Options, equals(['Red', 'Brown', 'White']));
        expect(provider.step2Concluded.value, isFalse);
      });

      test('derives autoAnswers count from auto_answered list length', () async {
        provider.step1Result.value = {
          'trait_extraction': {'visible_traits': <String, dynamic>{}},
        };
        fakeApi.step2Response = {
          'status': 'question',
          'session_id': 'sess-1',
          'question': 'Has it got gills?',
          'options': ['Yes', 'No'],
          'auto_answered': ['q0', 'q1', 'q2'],
          'path': ['q0', 'q1', 'q2', 'q3'],
        };

        await provider.runStep2Start();

        expect(provider.step2AutoAnswers.value, equals(3),
            reason: 'auto_answered list has 3 items');
        expect(provider.step2UserAnswers.value, equals(1),
            reason: 'path.length(4) - auto_answered.length(3) = 1');
      });

      test('prefers explicit autoAnswers field when present', () async {
        provider.step1Result.value = {
          'trait_extraction': {'visible_traits': <String, dynamic>{}},
        };
        fakeApi.step2Response = {
          'status': 'question',
          'session_id': 'sess-2',
          'question': 'Spore colour?',
          'options': ['White', 'Brown'],
          'auto_answers': 5,
          'user_answers': 2,
          'auto_answered': ['q0'],
          'path': ['q0'],
        };

        await provider.runStep2Start();

        expect(provider.step2AutoAnswers.value, equals(5),
            reason: 'explicit auto_answers field takes precedence');
        expect(provider.step2UserAnswers.value, equals(2),
            reason: 'explicit user_answers field takes precedence');
      });
    });

    group('_applyStep2Response — conclusion shape', () {
      test('sets step2Result, marks concluded, clears question/options', () async {
        provider.step1Result.value = {
          'trait_extraction': {'visible_traits': <String, dynamic>{}},
        };
        fakeApi.step2Response = {
          'status': 'conclusion',
          'session_id': 'sess-c',
          'species': 'Kantarell',
          'edibility': '*',
          'path': ['q0', 'q1'],
          'auto_answered': ['q0'],
        };

        await provider.runStep2Start();

        expect(provider.step2Concluded.value, isTrue);
        expect(provider.step2Result.value, isNotNull);
        expect(provider.step2Result.value!['species'], equals('Kantarell'));
        expect(provider.step2Question.value, isNull);
        expect(provider.step2Options, isEmpty);
        expect(provider.step2AutoAnswers.value, equals(1));
        expect(provider.step2UserAnswers.value, equals(1),
            reason: 'path.length(2) - auto_answered.length(1) = 1');
      });
    });

    group('runStep3AndStep4 sequencing', () {
      void _seedCompletedSteps() {
        provider.step1Result.value = {
          'trait_extraction': {
            'visible_traits': {'color': 'yellow'},
          },
        };
        provider.step2Result.value = {
          'status': 'conclusion',
          'species': 'Kantarell',
        };
        fakeApi.step3Response = {
          'trait_match': {'score': 0.9},
        };
        fakeApi.step4Response = {
          'final_recommendation': {
            'swedish_name': 'Kantarell',
            'overall_confidence': 0.88,
          },
        };
      }

      test('runStep3AndStep4 sets both step3Result and step4Result on success', () async {
        _seedCompletedSteps();
        await provider.runStep3AndStep4();

        expect(provider.step3Result.value, isNotNull);
        expect(provider.step4Result.value, isNotNull);
        expect(provider.errorMessage.value, isNull);
      });

      test('runStep3AndStep4 stops after Step 3 error and does not run Step 4', () async {
        _seedCompletedSteps();
        fakeApi.errorToThrow = Exception('Step 3 backend error');

        await provider.runStep3AndStep4();

        expect(provider.step3Result.value, isNull);
        expect(provider.step4Result.value, isNull);
        expect(provider.errorMessage.value, isNotNull);
      });

      test('runStep3AndStep4 fails early when Steps 1 or 2 are missing', () async {
        // No step1/step2 seeded
        await provider.runStep3AndStep4();

        expect(provider.errorMessage.value, isNotNull);
        expect(provider.step3Result.value, isNull);
        expect(provider.step4Result.value, isNull);
      });
    });
  });
}
