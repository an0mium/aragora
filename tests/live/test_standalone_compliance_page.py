from pathlib import Path


EU_AI_ACT_PAGE = Path("aragora/live/src/app/(standalone)/eu-ai-act/page.tsx")


def test_standalone_classifier_uses_shared_api_helper() -> None:
    source = EU_AI_ACT_PAGE.read_text(encoding="utf-8")

    assert "import { apiPost } from '@/lib/api';" in source
    assert "apiPost<ClassificationResponse>(" in source
    assert "'/api/v2/compliance/eu-ai-act/classify'" in source
    assert "fetch('/api/v2/compliance/eu-ai-act/classify'" not in source
