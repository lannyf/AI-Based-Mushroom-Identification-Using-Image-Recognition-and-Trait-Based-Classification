import 'package:get/get.dart';

class AppTranslations extends Translations {
  @override
  Map<String, Map<String, String>> get keys => {
        'en_US': _en,
        'sv_SE': _sv,
      };

  static const Map<String, String> _en = {
    // ── App ──────────────────────────────────────────────────────────────────
    'app_title': 'Mushroom ID',

    // ── Home page ────────────────────────────────────────────────────────────
    'identify_mushrooms': 'Identify Mushrooms',
    'home_subtitle': 'Use AI to identify mushroom species from photos',
    'take_photo': 'Take Photo',
    'view_history': 'View History',
    'recent_identifications': 'Recent Identifications',
    'recent_identifications_empty':
        'Your previous identifications will appear here',
    'safety_disclaimer': 'Safety Disclaimer',
    'safety_disclaimer_text':
        'This app is for educational and informational purposes only. '
            'Do NOT rely solely on this app for mushroom identification. '
            'Always consult with expert mycologists or your local poison control '
            'before consuming any wild mushroom.',

    // ── Camera page ──────────────────────────────────────────────────────────
    'capture_mushroom_image': 'Capture Mushroom Image',
    'preview_adjust': 'Preview & Adjust',
    'capture_title': 'Capture a Mushroom Image',
    'capture_desc':
        'Take a clear photo of the mushroom from above. '
            'Include the cap, gills (if visible), and stem for best results.',
    'photo_tips': 'Photography Tips',
    'tip_lighting': 'Use good natural lighting',
    'tip_center': 'Center the mushroom in frame',
    'tip_angles': 'Capture multiple angles',
    'tip_sharp': 'Keep image sharp',
    'tip_no_shadows': 'Avoid shadows and backlighting',
    'tip_leave_space': 'Leave some space around edges',
    'tip_show_parts': 'Show cap, gills, and stem if possible',
    'tip_hold_steady': 'Hold steady or use tripod',
    'gallery': 'Upload from Gallery',
    'reset': 'Reset',
    'rotate': 'Rotate',
    'retake': 'Retake',
    'continue_btn': 'Continue',
    'no_image_selected': 'No image selected',
    'image_too_large': 'Image is too large. Maximum size is 5 MB.',
    'failed_capture': 'Failed to capture photo',
    'failed_pick': 'Failed to pick image',
    'error_validating': 'Error validating image',

    // ── Questionnaire page ───────────────────────────────────────────────────
    'mushroom_details': 'Mushroom Details',
    'notes_optional': 'Additional Notes (Optional)',
    'notes_hint': 'Any other observations about the mushroom?',
    'select_one': 'Select one:',
    'identify_mushroom': 'Identify Mushroom',
    'next': 'Next',
    'previous': 'Previous',
    'skip': 'Skip',
    'missing_info': 'Missing Information',
    'missing_traits': 'Please select all required traits',
    'missing_image': 'Missing Image',
    'missing_image_desc': 'Please upload an image before continuing',
    'identification_failed': 'Identification Failed',

    // Questionnaire section titles & descriptions
    'cap_shape_title': 'Cap Shape',
    'cap_shape_desc': 'What shape is the mushroom cap?',
    'cap_color_title': 'Cap Color',
    'cap_color_desc': 'What color is the mushroom cap?',
    'gill_type_title': 'Gill Type',
    'gill_type_desc': 'How are the gills attached to the stem?',
    'stem_type_title': 'Stem Type',
    'stem_type_desc': 'What is the stem structure like?',
    'habitat_title': 'Habitat',
    'habitat_desc': 'Where did you find the mushroom?',
    'season_title': 'Season',
    'season_desc': 'When did you find the mushroom?',

    // Trait options – cap shape
    'trait_Convex': 'Convex',
    'trait_Flat': 'Flat',
    'trait_Conical': 'Conical',
    'trait_Wavy': 'Wavy',
    'trait_Bell-shaped': 'Bell-shaped',
    'trait_Umbrella-shaped': 'Umbrella-shaped',

    // Trait options – color
    'trait_White': 'White',
    'trait_Beige': 'Beige',
    'trait_Brown': 'Brown',
    'trait_Red': 'Red',
    'trait_Orange': 'Orange',
    'trait_Yellow': 'Yellow',
    'trait_Green': 'Green',
    'trait_Purple': 'Purple',
    'trait_Black': 'Black',
    'trait_Gray': 'Gray',

    // Trait options – gill type
    'trait_Free': 'Free',
    'trait_Attached': 'Attached',
    'trait_Decurrent': 'Decurrent',
    'trait_Subdecurrent': 'Subdecurrent',
    'trait_Crowded': 'Crowded',
    'trait_Sparse': 'Sparse',
    'trait_Forked': 'Forked',

    // Trait options – stem type
    'trait_Solid': 'Solid',
    'trait_Hollow': 'Hollow',
    'trait_Fibrous': 'Fibrous',
    'trait_Bulbous': 'Bulbous',
    'trait_Rooted': 'Rooted',
    'trait_Ring/Annulus': 'Ring / Annulus',
    'trait_Cup/Volva': 'Cup / Volva',

    // Trait options – habitat
    'trait_Forest': 'Forest',
    'trait_Grassland': 'Grassland',
    'trait_Garden': 'Garden',
    'trait_Dead wood': 'Dead wood',
    'trait_Living tree': 'Living tree',
    'trait_Underground': 'Underground',
    'trait_Disturbed soil': 'Disturbed soil',

    // Trait options – season
    'trait_Spring': 'Spring',
    'trait_Summer': 'Summer',
    'trait_Autumn': 'Autumn',
    'trait_Winter': 'Winter',

    // ── Results page ─────────────────────────────────────────────────────────
    'identification_results': 'Identification Results',
    'share_results': 'Share results',
    'save_to_history': 'Save to history',
    'try_again': 'Try Again',
    'save_result': 'Save Result',
    'back_to_home': 'Back to Home',
    'confidence': 'Confidence',
    'most_likely': 'Most likely identification',
    'confidence_by_method': 'Confidence by Method',
    'image_recognition': 'Image Recognition',
    'trait_analysis': 'Trait Analysis',
    'language_model': 'Language Model',
    'top_predictions_label': 'Top Predictions',
    'lookalike_warning': 'Lookalike Warning',
    'edible': 'Edible',
    'toxic': 'Toxic',
    'unknown': 'Unknown',
    'save': 'Save',
    'share': 'Share',

    // Safety ratings
    'likely_edible': 'Likely Edible',
    'likely_edible_desc':
        'This species is generally recognized as edible if properly identified.',
    'caution_advised': 'Caution Advised',
    'caution_desc':
        'This species requires careful identification. '
            'Some similar species may be toxic.',
    'not_recommended': 'NOT Recommended',
    'not_recommended_desc': 'This species is toxic or inedible. Do not consume.',
    'safety_unknown': 'Safety Unknown',
    'safety_unknown_desc':
        'Insufficient data. Do not consume without expert verification.',
    'safety_unavailable': 'Safety information unavailable',

    // Safety notice
    'important_safety_notice': 'Important Safety Notice',
    'safety_notice_text':
        'This app is for educational and informational purposes only. '
            'DO NOT rely solely on this app for mushroom identification or consumption decisions. '
            'Always consult with:',
    'safety_expert_mycologists': 'Expert mycologists',
    'safety_expert_poison': 'Local poison control centers',
    'safety_expert_guides': 'Multiple field guides',

    // Snackbars – results
    'share_coming_soon': 'Result sharing feature coming soon',
    'save_failed': 'Save Failed',
    'no_image_path': 'No image path available for this result',
    'saved': 'Saved',
    'result_saved': 'Result saved to history',

    // Demo mode banner
    'demo_notice':
        'Demo results only: this screen is showing hardcoded sample data. '
            'It is not yet using the uploaded image for real identification.',

    // Mushroom common names (demo data)
    'mushroom_Porcini': 'Porcini',
    'mushroom_Summer Porcini': 'Summer Porcini',
    'mushroom_Red-Foot Bolete': 'Red-Foot Bolete',
    'mushroom_King Bolete': 'King Bolete',
    'mushroom_Slippery Jack': 'Slippery Jack',

    // Lookalike reasons (demo data)
    'lookalike_reason_calopus': 'Inedible, can cause stomach upset',
    'lookalike_reason_sensibilis': 'Mild toxin, similar appearance',

    // ── History page ─────────────────────────────────────────────────────────
    'identification_history': 'Identification History',
    'no_history': 'No identifications yet',
    'no_history_subtitle': 'Start by taking a photo of a mushroom',
    'clear_history_tooltip': 'Clear history',
    'delete_entry_title': 'Delete Entry?',
    'delete_entry_confirm':
        'Are you sure you want to delete this identification?',
    'identification_deleted': 'Identification deleted',
    'failed_delete': 'Failed to delete identification',
    'clear_history_title': 'Clear History?',
    'clear_history_confirm':
        'This will permanently delete all saved identifications. '
            'This action cannot be undone.',
    'history_cleared': 'History cleared',
    'failed_clear': 'Failed to clear history',
    'delete': 'Delete',
    'view_label': 'View',
    'identification_details_title': 'Identification Details',
    'species_identified': 'Species Identified',
    'safety_label': 'Safety',
    'traits_selected': 'Traits Selected',
    'notes_label': 'Notes',
    'no_traits_recorded': 'No traits recorded',
    'share_coming_soon_history': 'Share functionality coming soon',
    'full_results_data': 'Full Results Data',
    'image_label': 'Image',

    // ── Settings page ────────────────────────────────────────────────────────
    'settings': 'Settings',
    'app_settings': 'App Settings',
    'enable_notifications': 'Enable Notifications',
    'notifications_subtitle': 'Get alerts for identification results',
    'language': 'Language',
    'language_subtitle': 'Select app language',
    'dark_mode': 'Dark Mode',
    'dark_mode_subtitle': 'Use dark theme (beta)',
    'storage_data': 'Storage & Data',
    'clear_cache': 'Clear Cache',
    'clear_cache_subtitle': 'Remove cached images and data',
    'export_history': 'Export History',
    'export_subtitle': 'Export identification history as JSON',
    'clear_all_data': 'Clear All Data',
    'clear_all_subtitle': 'Permanently delete all saved identifications',
    'developer_tools': 'Developer Tools',
    'view_logs': 'View Logs',
    'view_logs_subtitle': 'View application logs',
    'database_info': 'Database Info',
    'database_info_subtitle': 'View database statistics',
    'app_version': 'App Version',
    'build_number': 'Build 1',
    'privacy_policy': 'Privacy Policy',
    'terms_of_service': 'Terms of Service',
    'about': 'About',
    'about_subtitle': 'AI-based Mushroom Identification System',
    'debug_mode_on': 'Debug Mode: ON',
    'debug_mode_off': 'Debug Mode: OFF',

    // Settings dialogs
    'connection_test_title': 'Connection Test',
    'connection_test_content': 'Testing connection to API server...',
    'clear_cache_confirm_title': 'Clear Cache?',
    'clear_cache_confirm_text':
        'This will remove cached images and data. '
            'Your identification history will not be affected.',
    'cache_cleared': 'Cache cleared',
    'clear_all_data_title': 'Delete All Data?',
    'clear_all_data_text':
        'This will permanently delete all saved identifications, '
            'history, and preferences. This action cannot be undone.',
    'all_data_deleted': 'All data deleted',
    'export_coming_soon': 'Export functionality coming soon',
    'application_logs_title': 'Application Logs',
    'logs_placeholder':
        'Logs would be displayed here.\nFeature in development.',
    'database_information_title': 'Database Information',
    'database_placeholder':
        'Database: mushroom_identification.db\n'
            'Tables: history, preferences\nFeature in development.',
    'privacy_policy_text':
        'We do not store images longer than necessary for processing.\n\n'
            'Your identification history is stored locally on your device.',
    'terms_text':
        'This app is provided "as is" without warranty.\n\n'
            'Identifications provided by this app are for reference only. '
            'Always verify with experts before consuming any mushroom.\n\n'
            'We are not liable for any consequences from using this app.',
    'about_app_name': 'Mushroom Identification System',
    'about_app_desc':
        'An AI-powered application that uses image recognition and machine '
            'learning to identify mushroom species.',
    'about_features': 'Features:',
    'about_features_list':
        '• Image-based identification\n'
            '• Trait-based classification\n'
            '• Species safety ratings\n'
            '• Identification history',
    'error_saving_setting': 'Error saving setting',
    'save_history': 'Save History',
    'save_history_subtitle': 'Automatically save identification results',
    'high_quality_images': 'High Quality Images',
    'high_quality_subtitle': 'Use higher resolution for better accuracy',
    'api_settings': 'API Settings',
    'test_connection': 'Test Connection',
    'version': 'Version',

    // ── Common ───────────────────────────────────────────────────────────────
    'ok': 'OK',
    'cancel': 'Cancel',
    'close': 'Close',
    'error': 'Error',
    'loading': 'Loading...',
    'retry': 'Retry',
    'back': 'Back',
    'delete_all': 'Delete All',
    'clear': 'Clear',
    'clear_all': 'Clear All',
    'additional_notes': 'Additional Notes',
  };

  static const Map<String, String> _sv = {
    // ── App ──────────────────────────────────────────────────────────────────
    'app_title': 'Svamp ID',

    // ── Home page ────────────────────────────────────────────────────────────
    'identify_mushrooms': 'Identifiera Svampar',
    'home_subtitle': 'Använd AI för att identifiera svamparter från foton',
    'take_photo': 'Ta Foto',
    'view_history': 'Visa Historik',
    'recent_identifications': 'Senaste Identifieringar',
    'recent_identifications_empty':
        'Dina tidigare identifieringar visas här',
    'safety_disclaimer': 'Säkerhetsfriskrivning',
    'safety_disclaimer_text':
        'Denna app är endast för utbildnings- och informationsändamål. '
            'Förlita dig INTE enbart på denna app för svampidentifiering. '
            'Rådgör alltid med erfarna mykologer eller din lokala giftinformation '
            'innan du äter någon vild svamp.',

    // ── Camera page ──────────────────────────────────────────────────────────
    'capture_mushroom_image': 'Ta Bild på Svamp',
    'preview_adjust': 'Förhandsgranska & Justera',
    'capture_title': 'Ta en Bild på en Svamp',
    'capture_desc':
        'Ta ett tydligt foto av svampen uppifrån. '
            'Inkludera hatten, lamellerna (om synliga) och skaftet för bästa resultat.',
    'photo_tips': 'Fototips',
    'tip_lighting': 'Använd bra naturligt ljus',
    'tip_center': 'Centrera svampen i bilden',
    'tip_angles': 'Ta bilder från flera vinklar',
    'tip_sharp': 'Håll bilden skarp',
    'tip_no_shadows': 'Undvik skuggor och motljus',
    'tip_leave_space': 'Lämna lite utrymme runt kanterna',
    'tip_show_parts': 'Visa hatt, lameller och skaft om möjligt',
    'tip_hold_steady': 'Håll stadigt eller använd stativ',
    'gallery': 'Ladda upp från Galleri',
    'reset': 'Återställ',
    'rotate': 'Rotera',
    'retake': 'Ta Om',
    'continue_btn': 'Fortsätt',
    'no_image_selected': 'Ingen bild vald',
    'image_too_large': 'Bilden är för stor. Maximal storlek är 5 MB.',
    'failed_capture': 'Kunde inte ta foto',
    'failed_pick': 'Kunde inte välja bild',
    'error_validating': 'Fel vid validering av bild',

    // ── Questionnaire page ───────────────────────────────────────────────────
    'mushroom_details': 'Svampdetaljer',
    'notes_optional': 'Ytterligare anteckningar (valfritt)',
    'notes_hint': 'Övriga observationer om svampen?',
    'select_one': 'Välj ett:',
    'identify_mushroom': 'Identifiera Svamp',
    'next': 'Nästa',
    'previous': 'Föregående',
    'skip': 'Hoppa över',
    'missing_info': 'Information saknas',
    'missing_traits': 'Välj alla obligatoriska egenskaper',
    'missing_image': 'Bild saknas',
    'missing_image_desc': 'Ladda upp en bild innan du fortsätter',
    'identification_failed': 'Identifiering misslyckades',

    // Questionnaire section titles & descriptions
    'cap_shape_title': 'Hattform',
    'cap_shape_desc': 'Vilken form har svamphatten?',
    'cap_color_title': 'Hattfärg',
    'cap_color_desc': 'Vilken färg har svamphatten?',
    'gill_type_title': 'Lamelltyp',
    'gill_type_desc': 'Hur sitter lamellerna fast vid skaftet?',
    'stem_type_title': 'Skafttyp',
    'stem_type_desc': 'Hur ser skaftstrukturen ut?',
    'habitat_title': 'Habitat',
    'habitat_desc': 'Var hittade du svampen?',
    'season_title': 'Säsong',
    'season_desc': 'När hittade du svampen?',

    // Trait options – cap shape
    'trait_Convex': 'Konvex',
    'trait_Flat': 'Platt',
    'trait_Conical': 'Konisk',
    'trait_Wavy': 'Vågig',
    'trait_Bell-shaped': 'Klocklikt formad',
    'trait_Umbrella-shaped': 'Paraplyformad',

    // Trait options – color
    'trait_White': 'Vit',
    'trait_Beige': 'Beige',
    'trait_Brown': 'Brun',
    'trait_Red': 'Röd',
    'trait_Orange': 'Orange',
    'trait_Yellow': 'Gul',
    'trait_Green': 'Grön',
    'trait_Purple': 'Lila',
    'trait_Black': 'Svart',
    'trait_Gray': 'Grå',

    // Trait options – gill type
    'trait_Free': 'Fria',
    'trait_Attached': 'Fastväxande',
    'trait_Decurrent': 'Nedlöpande',
    'trait_Subdecurrent': 'Lätt nedlöpande',
    'trait_Crowded': 'Täta',
    'trait_Sparse': 'Glesa',
    'trait_Forked': 'Gafflade',

    // Trait options – stem type
    'trait_Solid': 'Solid',
    'trait_Hollow': 'Ihålig',
    'trait_Fibrous': 'Fiberaktig',
    'trait_Bulbous': 'Knölig',
    'trait_Rooted': 'Rotad',
    'trait_Ring/Annulus': 'Ring / Annulus',
    'trait_Cup/Volva': 'Kopp / Volva',

    // Trait options – habitat
    'trait_Forest': 'Skog',
    'trait_Grassland': 'Gräsmark',
    'trait_Garden': 'Trädgård',
    'trait_Dead wood': 'Dött trä',
    'trait_Living tree': 'Levande träd',
    'trait_Underground': 'Under jord',
    'trait_Disturbed soil': 'Störd jord',

    // Trait options – season
    'trait_Spring': 'Vår',
    'trait_Summer': 'Sommar',
    'trait_Autumn': 'Höst',
    'trait_Winter': 'Vinter',

    // ── Results page ─────────────────────────────────────────────────────────
    'identification_results': 'Identifieringsresultat',
    'share_results': 'Dela resultat',
    'save_to_history': 'Spara i historik',
    'try_again': 'Försök igen',
    'save_result': 'Spara resultat',
    'back_to_home': 'Tillbaka till start',
    'confidence': 'Säkerhet',
    'most_likely': 'Mest trolig identifiering',
    'confidence_by_method': 'Säkerhet per metod',
    'image_recognition': 'Bildidentifiering',
    'trait_analysis': 'Egenskapsanalys',
    'language_model': 'Språkmodell',
    'top_predictions_label': 'Bästa förutsägelser',
    'lookalike_warning': 'Varning för förväxlingsbara arter',
    'edible': 'Ätlig',
    'toxic': 'Giftig',
    'unknown': 'Okänd',
    'save': 'Spara',
    'share': 'Dela',

    // Safety ratings
    'likely_edible': 'Troligen ätlig',
    'likely_edible_desc':
        'Denna art är allmänt erkänd som ätlig om den identifieras korrekt.',
    'caution_advised': 'Försiktighet rekommenderas',
    'caution_desc':
        'Denna art kräver noggrann identifiering. '
            'Vissa liknande arter kan vara giftiga.',
    'not_recommended': 'REKOMMENDERAS EJ',
    'not_recommended_desc':
        'Denna art är giftig eller oätlig. Ät inte.',
    'safety_unknown': 'Säkerhet okänd',
    'safety_unknown_desc':
        'Otillräckliga data. Ät inte utan expertkontroll.',
    'safety_unavailable': 'Säkerhetsinformation saknas',

    // Safety notice
    'important_safety_notice': 'Viktig säkerhetsinformation',
    'safety_notice_text':
        'Denna app är för utbildnings- och informationsändamål. '
            'Förlita dig INTE på appen för beslut om att äta svamp. '
            'Rådfråga alltid:',
    'safety_expert_mycologists': 'Expertmykologer',
    'safety_expert_poison': 'Lokala giftinformationscentraler',
    'safety_expert_guides': 'Flera fältguider',

    // Snackbars – results
    'share_coming_soon': 'Delningsfunktionen kommer snart',
    'save_failed': 'Kunde inte spara',
    'no_image_path': 'Ingen bildsökväg tillgänglig för detta resultat',
    'saved': 'Sparat',
    'result_saved': 'Resultatet sparades i historiken',

    // Demo mode banner
    'demo_notice':
        'Demoläge: den här skärmen visar förkodade exempeldata. '
            'Den verkliga bilden används ännu inte för identifiering.',

    // Mushroom common names (demo data) – Swedish names
    'mushroom_Porcini': 'Karljohansvamp',
    'mushroom_Summer Porcini': 'Sommarsopp',
    'mushroom_Red-Foot Bolete': 'Rävskogssopp',
    'mushroom_King Bolete': 'Kungsopp',
    'mushroom_Slippery Jack': 'Gul slidsopp',

    // Lookalike reasons (demo data)
    'lookalike_reason_calopus': 'Oätlig, kan orsaka magbesvär',
    'lookalike_reason_sensibilis': 'Svagt toxin, liknande utseende',

    // ── History page ─────────────────────────────────────────────────────────
    'identification_history': 'Identifieringshistorik',
    'no_history': 'Inga identifieringar ännu',
    'no_history_subtitle': 'Börja med att ta ett foto av en svamp',
    'clear_history_tooltip': 'Rensa historik',
    'delete_entry_title': 'Radera post?',
    'delete_entry_confirm':
        'Är du säker på att du vill radera denna identifiering?',
    'identification_deleted': 'Identifiering raderad',
    'failed_delete': 'Det gick inte att radera identifieringen',
    'clear_history_title': 'Rensa historik?',
    'clear_history_confirm':
        'Detta raderar permanent alla sparade identifieringar. '
            'Åtgärden kan inte ångras.',
    'history_cleared': 'Historik rensad',
    'failed_clear': 'Det gick inte att rensa historiken',
    'delete': 'Radera',
    'view_label': 'Visa',
    'identification_details_title': 'Identifieringsdetaljer',
    'species_identified': 'Identifierad art',
    'safety_label': 'Säkerhet',
    'traits_selected': 'Valda egenskaper',
    'notes_label': 'Anteckningar',
    'no_traits_recorded': 'Inga egenskaper registrerade',
    'share_coming_soon_history': 'Delningsfunktionen kommer snart',
    'full_results_data': 'Fullständiga resultatdata',
    'image_label': 'Bild',

    // ── Settings page ────────────────────────────────────────────────────────
    'settings': 'Inställningar',
    'app_settings': 'Appinställningar',
    'enable_notifications': 'Aktivera aviseringar',
    'notifications_subtitle': 'Få aviseringar för identifieringsresultat',
    'language': 'Språk',
    'language_subtitle': 'Välj appspråk',
    'dark_mode': 'Mörkt läge',
    'dark_mode_subtitle': 'Använd mörkt tema (beta)',
    'storage_data': 'Lagring och data',
    'clear_cache': 'Rensa cache',
    'clear_cache_subtitle': 'Ta bort cachade bilder och data',
    'export_history': 'Exportera historik',
    'export_subtitle': 'Exportera identifieringshistorik som JSON',
    'clear_all_data': 'Rensa all data',
    'clear_all_subtitle': 'Radera alla sparade identifieringar permanent',
    'developer_tools': 'Utvecklarverktyg',
    'view_logs': 'Visa loggar',
    'view_logs_subtitle': 'Visa applikationsloggar',
    'database_info': 'Databasinformation',
    'database_info_subtitle': 'Visa databasstatistik',
    'app_version': 'Appversion',
    'build_number': 'Version 1',
    'privacy_policy': 'Integritetspolicy',
    'terms_of_service': 'Användarvillkor',
    'about': 'Om',
    'about_subtitle': 'AI-baserat svampidentifieringssystem',
    'debug_mode_on': 'Felsökningsläge: PÅ',
    'debug_mode_off': 'Felsökningsläge: AV',

    // Settings dialogs
    'connection_test_title': 'Anslutningstest',
    'connection_test_content': 'Testar anslutning till API-server...',
    'clear_cache_confirm_title': 'Rensa cache?',
    'clear_cache_confirm_text':
        'Detta tar bort cachade bilder och data. '
            'Din identifieringshistorik påverkas inte.',
    'cache_cleared': 'Cache rensad',
    'clear_all_data_title': 'Radera all data?',
    'clear_all_data_text':
        'Detta raderar permanent alla sparade identifieringar, '
            'historik och inställningar. Åtgärden kan inte ångras.',
    'all_data_deleted': 'All data raderad',
    'export_coming_soon': 'Exportfunktion kommer snart',
    'application_logs_title': 'Applikationsloggar',
    'logs_placeholder':
        'Loggar visas här.\nFunktion under utveckling.',
    'database_information_title': 'Databasinformation',
    'database_placeholder':
        'Databas: mushroom_identification.db\n'
            'Tabeller: historik, inställningar\nFunktion under utveckling.',
    'privacy_policy_text':
        'Vi lagrar inte bilder längre än nödvändigt för bearbetning.\n\n'
            'Din identifieringshistorik lagras lokalt på din enhet.',
    'terms_text':
        'Denna app tillhandahålls i befintligt skick utan garanti.\n\n'
            'Identifieringar i appen är endast vägledande. '
            'Verifiera alltid med experter innan du äter svamp.\n\n'
            'Vi ansvarar inte för konsekvenser av att använda appen.',
    'about_app_name': 'Svampidentifieringssystem',
    'about_app_desc':
        'En AI-driven applikation som använder bildigenkänning och '
            'maskininlärning för att identifiera svamparter.',
    'about_features': 'Funktioner:',
    'about_features_list':
        '• Bildbaserad identifiering\n'
            '• Egenskapsbaserad klassificering\n'
            '• Säkerhetsbetyg för arter\n'
            '• Identifieringshistorik',
    'error_saving_setting': 'Fel vid sparande av inställning',
    'save_history': 'Spara historik',
    'save_history_subtitle': 'Spara identifieringsresultat automatiskt',
    'high_quality_images': 'Högkvalitativa bilder',
    'high_quality_subtitle': 'Använd högre upplösning för bättre noggrannhet',
    'api_settings': 'API-inställningar',
    'test_connection': 'Testa anslutning',
    'version': 'Version',

    // ── Common ───────────────────────────────────────────────────────────────
    'ok': 'OK',
    'cancel': 'Avbryt',
    'close': 'Stäng',
    'error': 'Fel',
    'loading': 'Laddar...',
    'retry': 'Försök igen',
    'back': 'Tillbaka',
    'delete_all': 'Radera allt',
    'clear': 'Rensa',
    'clear_all': 'Rensa allt',
    'additional_notes': 'Ytterligare anteckningar',
  };
}
