"""Tests for canvas/primitives.py — A2UI primitives."""

from aragora.canvas.primitives import (
    AlertPrimitive,
    AlertSeverity,
    ButtonPrimitive,
    ButtonVariant,
    CardPrimitive,
    ChartDataPoint,
    ChartPrimitive,
    ChartSeries,
    ChartType,
    FormField,
    FormPrimitive,
    ProgressPrimitive,
    SelectOption,
    SelectPrimitive,
    TableColumn,
    TablePrimitive,
)


class TestEnums:
    def test_button_variants(self):
        assert ButtonVariant.PRIMARY.value == "primary"
        assert ButtonVariant.DANGER.value == "danger"
        assert ButtonVariant.GHOST.value == "ghost"

    def test_alert_severity(self):
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.ERROR.value == "error"

    def test_chart_type(self):
        assert ChartType.LINE.value == "line"
        assert ChartType.PIE.value == "pie"
        assert ChartType.SCATTER.value == "scatter"


class TestButtonPrimitive:
    def test_defaults(self):
        b = ButtonPrimitive(label="Click", action="do_thing")
        assert b.variant == ButtonVariant.PRIMARY
        assert b.disabled is False
        assert b.loading is False

    def test_to_dict(self):
        b = ButtonPrimitive(label="Run", action="run_analysis", variant=ButtonVariant.SUCCESS)
        d = b.to_dict()
        assert d["type"] == "button"
        assert d["label"] == "Run"
        assert d["variant"] == "success"
        assert d["metadata"] == {}


class TestFormField:
    def test_defaults(self):
        f = FormField(name="email", field_type="email", label="Email")
        assert f.required is False
        assert f.options == []

    def test_to_dict(self):
        f = FormField(name="name", field_type="text", label="Name", required=True)
        d = f.to_dict()
        assert d["name"] == "name"
        assert d["type"] == "text"
        assert d["required"] is True


class TestFormPrimitive:
    def test_to_dict(self):
        fields = [FormField(name="q", field_type="text", label="Question")]
        form = FormPrimitive(fields=fields, submit_action="submit_form", title="My Form")
        d = form.to_dict()
        assert d["type"] == "form"
        assert len(d["fields"]) == 1
        assert d["submit_action"] == "submit_form"
        assert d["title"] == "My Form"


class TestSelectPrimitive:
    def test_defaults(self):
        opts = [SelectOption(value="a", label="A"), SelectOption(value="b", label="B")]
        s = SelectPrimitive(options=opts, action="choose")
        assert s.placeholder == "Select..."
        assert s.multiple is False

    def test_to_dict(self):
        opts = [SelectOption(value="x", label="X", disabled=True)]
        s = SelectPrimitive(options=opts, action="pick", searchable=True)
        d = s.to_dict()
        assert d["type"] == "select"
        assert d["options"][0]["disabled"] is True
        assert d["searchable"] is True


class TestChartPrimitive:
    def test_to_dict(self):
        series = [
            ChartSeries(
                name="Revenue",
                data=[ChartDataPoint(x="Jan", y=100.0), ChartDataPoint(x="Feb", y=150.0)],
                color="blue",
            )
        ]
        chart = ChartPrimitive(chart_type=ChartType.BAR, series=series, title="Revenue")
        d = chart.to_dict()
        assert d["type"] == "chart"
        assert d["chart_type"] == "bar"
        assert len(d["series"]) == 1
        assert d["series"][0]["data"][0]["y"] == 100.0
        assert d["title"] == "Revenue"
        assert d["show_legend"] is True


class TestProgressPrimitive:
    def test_defaults(self):
        p = ProgressPrimitive(value=50.0)
        assert p.max_value == 100.0
        assert p.animated is True

    def test_percentage(self):
        p = ProgressPrimitive(value=30, max_value=60)
        assert p.percentage == 50.0

    def test_percentage_zero_max(self):
        p = ProgressPrimitive(value=10, max_value=0)
        assert p.percentage == 0.0

    def test_percentage_capped(self):
        p = ProgressPrimitive(value=200, max_value=100)
        assert p.percentage == 100.0

    def test_to_dict(self):
        p = ProgressPrimitive(value=75, label="Loading...")
        d = p.to_dict()
        assert d["type"] == "progress"
        assert d["percentage"] == 75.0
        assert d["label"] == "Loading..."


class TestAlertPrimitive:
    def test_defaults(self):
        a = AlertPrimitive(message="Hello")
        assert a.severity == AlertSeverity.INFO
        assert a.dismissible is True

    def test_to_dict(self):
        a = AlertPrimitive(
            message="Error!", severity=AlertSeverity.ERROR, title="Oops", auto_dismiss_seconds=5
        )
        d = a.to_dict()
        assert d["type"] == "alert"
        assert d["severity"] == "error"
        assert d["auto_dismiss_seconds"] == 5


class TestCardPrimitive:
    def test_defaults(self):
        c = CardPrimitive(title="Card", content="Body")
        assert c.actions == []
        assert c.image_url is None

    def test_to_dict_with_actions(self):
        btn = ButtonPrimitive(label="Open", action="open")
        c = CardPrimitive(title="Card", content="Body", actions=[btn])
        d = c.to_dict()
        assert d["type"] == "card"
        assert len(d["actions"]) == 1
        assert d["actions"][0]["type"] == "button"


class TestTablePrimitive:
    def test_to_dict(self):
        cols = [TableColumn(key="name", header="Name", sortable=True)]
        rows = [{"name": "Alice"}, {"name": "Bob"}]
        t = TablePrimitive(columns=cols, rows=rows, title="Users")
        d = t.to_dict()
        assert d["type"] == "table"
        assert len(d["columns"]) == 1
        assert d["columns"][0]["sortable"] is True
        assert len(d["rows"]) == 2
        assert d["title"] == "Users"

    def test_defaults(self):
        t = TablePrimitive(columns=[], rows=[])
        assert t.sortable is True
        assert t.paginated is False
        assert t.page_size == 10
