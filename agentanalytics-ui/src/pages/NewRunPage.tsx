import React, { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Divider,
  Form,
  Input,
  InputNumber,
  Select,
  Switch,
  Tabs,
  Typography,
  Upload,
  message
} from "antd";
import { UploadOutlined } from "@ant-design/icons";
import Editor from "@monaco-editor/react";
import yaml from "js-yaml";
import { createRunFromYaml, listPlugins, PluginMeta } from "../lib/api";
import { useNavigate } from "react-router-dom";

const LS_YAML = "aa.newrun.yaml";
const LS_FORM = "aa.newrun.form";
const LS_TAB = "aa.newrun.tab";

export default function NewRunPage() {
  const nav = useNavigate();
  const [plugins, setPlugins] = useState<PluginMeta[]>([]);

  const [yamlText, setYamlText] = useState<string>(() => {
    return localStorage.getItem(LS_YAML) || DEFAULT_YAML;
  });
  const [activeTab, setActiveTab] = useState<string>(() => {
    return localStorage.getItem(LS_TAB) || "yaml";
  });
  const [hydrated, setHydrated] = useState(false);

  const [submitting, setSubmitting] = useState(false);

  // Form state
  const [form] = Form.useForm();

  // Load plugin metadata
  useEffect(() => {
    (async () => {
      try {
        const metas = await listPlugins();
        setPlugins(metas);
      } catch (e) {
        console.error(e);
      }
    })();
  }, []);

  useEffect(() => {
    const savedForm = localStorage.getItem(LS_FORM);
    if (savedForm) {
      try {
        form.setFieldsValue(JSON.parse(savedForm));
      } catch {
        // ignore
      }
    }
    setHydrated(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(LS_YAML, yamlText);
  }, [yamlText, hydrated]);

  useEffect(() => {
    if (!hydrated) return;
    localStorage.setItem(LS_TAB, activeTab);
  }, [activeTab, hydrated]);

  const parsedYamlError = useMemo(() => {
    try {
      yaml.load(yamlText);
      return null;
    } catch (e: any) {
      return e?.message || "Invalid YAML";
    }
  }, [yamlText]);

  async function onSubmitYaml() {
    if (parsedYamlError) {
      message.error("Fix YAML errors before submitting.");
      return;
    }
    setSubmitting(true);
    try {
      const r = await createRunFromYaml(yamlText);
      message.success(`Run started: ${r.run_id}`);
      nav(`/runs/${r.run_id}`);
    } finally {
      setSubmitting(false);
    }
  }

  function onUpload(file: File) {
    const reader = new FileReader();
    reader.onload = () => {
      const text = String(reader.result || "");
      setYamlText(text);
      setActiveTab("yaml");
      message.success("Loaded YAML");

      // Sync YAML -> Form (if possible)
      try {
        const vals = yamlToFormValues(text, plugins);
        form.setFieldsValue(vals);
        localStorage.setItem(LS_FORM, JSON.stringify(form.getFieldsValue(true)));
      } catch {
        // ignore parse errors
      }
    };
    reader.readAsText(file);
    return false; // prevent upload
  }

  function applyYamlToForm() {
    if (parsedYamlError) {
      message.error("Invalid YAML. Fix errors before applying to the form.");
      return;
    }
    try {
      const vals = yamlToFormValues(yamlText, plugins);
      form.setFieldsValue(vals);
      localStorage.setItem(LS_FORM, JSON.stringify(form.getFieldsValue(true)));
      message.success("Applied YAML to Form");
      setActiveTab("form");
    } catch {
      message.error("Could not apply YAML to Form.");
    }
  }

  function applyFormToYaml(values: any) {
    // Build config JSON from widgets; then dump to YAML
    const config: any = {
      mlflow: { tracking_uri: values.tracking_uri },
      timeframe: {},
      output: { output_dir: values.output_dir || "./artifacts", overwrite: true },
      schema_bindings: {},
      plugins: (plugins || []).map((p) => ({
        name: p.name,
        enabled: !!values?.plugins?.[p.name]?.enabled,
        config: { ...(p.defaults || {}), ...(values?.plugins?.[p.name]?.config || {}) }
      }))
    };

    // timeframe selection
    if (values.timeframe_mode === "days") config.timeframe.last_n_days = values.last_n_days ?? 7;
    if (values.timeframe_mode === "hours") config.timeframe.last_n_hours = values.last_n_hours ?? 24;
    if (values.timeframe_mode === "traces") config.timeframe.last_n_traces = values.last_n_traces ?? 1000;

    const y = yaml.dump(config, { sortKeys: false });
    setYamlText(y);
    message.success("Updated YAML from form");
    setActiveTab("yaml");
  }

  return (
    <Card
      title="Create Run"
      extra={
        <Button
          type="primary"
          onClick={onSubmitYaml}
          loading={submitting}
          disabled={!!parsedYamlError}
        >
          Run
        </Button>
      }
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: "load",
            label: "Load YAML",
            children: (
              <>
                <Typography.Paragraph>
                  Upload a YAML config. It will populate the editor (and the Form tab).
                </Typography.Paragraph>
                <Upload beforeUpload={onUpload} showUploadList={false}>
                  <Button type="primary" icon={<UploadOutlined />}>Choose YAML file</Button>
                </Upload>
              </>
            )
          },
          {
            key: "yaml",
            label: "YAML Editor",
            children: (
              <>
                {parsedYamlError ? (
                  <Alert type="error" message="YAML Error" description={parsedYamlError} />
                ) : (
                  <Alert type="success" message="YAML looks valid" />
                )}
                <Divider />

                <Editor
                  height="520px"
                  defaultLanguage="yaml"
                  theme="vs-dark"
                  value={yamlText}
                  onChange={(v) => setYamlText(v ?? "")}
                  options={{
                    minimap: { enabled: false },
                    fontSize: 13,
                    wordWrap: "on",
                    scrollBeyondLastLine: false,
                    automaticLayout: true
                  }}
                />

                <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
                  <Button type="primary" onClick={applyYamlToForm} disabled={!!parsedYamlError || plugins.length === 0}>
                    Apply YAML to Form
                  </Button>
                  <Typography.Text type="secondary">
                    (Tip: Use this after manual YAML edits)
                  </Typography.Text>
                </div>
              </>
            )
          },
          {
            key: "form",
            label: "Form",
            children: (
              <>
                <Alert
                  type="info"
                  message="Form view"
                  description="Fill settings using widgets. Click “Generate YAML” to update the YAML editor."
                />
                <Divider />
                <Form
                  form={form}
                  layout="vertical"
                  initialValues={{
                    tracking_uri: "http://localhost:5000",
                    output_dir: "./artifacts",
                    timeframe_mode: "days",
                    last_n_days: 7,
                    plugins: {}
                  }}
                  onFinish={applyFormToYaml}
                  onValuesChange={() => {
                    const v = form.getFieldsValue(true);
                    localStorage.setItem(LS_FORM, JSON.stringify(v));
                  }}
                >
                  <Form.Item label="MLflow tracking URI" name="tracking_uri" rules={[{ required: true }]}>
                    <Input />
                  </Form.Item>

                  <Form.Item label="Output directory" name="output_dir">
                    <Input />
                  </Form.Item>

                  <Divider />
                  <Typography.Title level={5}>Timeframe</Typography.Title>

                  <Form.Item label="Mode" name="timeframe_mode">
                    <Select
                      options={[
                        { label: "Last N days", value: "days" },
                        { label: "Last N hours", value: "hours" },
                        { label: "Last N traces", value: "traces" }
                      ]}
                    />
                  </Form.Item>

                  <Form.Item noStyle shouldUpdate={(p, c) => p.timeframe_mode !== c.timeframe_mode}>
                    {({ getFieldValue }) => {
                      const mode = getFieldValue("timeframe_mode");
                      if (mode === "hours") {
                        return (
                          <Form.Item label="Last N hours" name="last_n_hours">
                            <InputNumber min={1} max={24 * 365} style={{ width: 200 }} />
                          </Form.Item>
                        );
                      }
                      if (mode === "traces") {
                        return (
                          <Form.Item label="Last N traces" name="last_n_traces">
                            <InputNumber min={10} max={1_000_000} style={{ width: 200 }} />
                          </Form.Item>
                        );
                      }
                      return (
                        <Form.Item label="Last N days" name="last_n_days">
                          <InputNumber min={1} max={3650} style={{ width: 200 }} />
                        </Form.Item>
                      );
                    }}
                  </Form.Item>

                  <Divider />
                  <Typography.Title level={5}>Plugins</Typography.Title>

                  {plugins.length === 0 ? (
                    <Alert type="warning" message="No plugins found (is the server running?)" />
                  ) : (
                    plugins.map((p) => (
                      <Card key={p.name} size="small" style={{ marginBottom: 12 }}>
                        <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
                          <div>
                            <Typography.Text strong>{p.title}</Typography.Text>
                            <div style={{ color: "#888" }}>{p.description}</div>
                            <div style={{ color: "#aaa", fontSize: 12 }}>
                              {p.name} v{p.version}
                            </div>
                          </div>
                          <Form.Item
                            label="Enabled"
                            name={["plugins", p.name, "enabled"]}
                            valuePropName="checked"
                            style={{ marginBottom: 0 }}
                          >
                            <Switch />
                          </Form.Item>
                        </div>

                        <Divider style={{ margin: "12px 0" }} />

                        <Typography.Text type="secondary">Settings</Typography.Text>

                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                            gap: 12,
                            marginTop: 10
                          }}
                        >
                          {p.params.map((param) => (
                            <ParamField key={param.key} pluginName={p.name} param={param} />
                          ))}
                        </div>
                      </Card>
                    ))
                  )}

                  <Divider />
                  <Button htmlType="submit" type="primary">
                    Generate YAML
                  </Button>
                </Form>
              </>
            )
          }
        ]}
      />
    </Card>
  );
}

function ParamField({ pluginName, param }: { pluginName: string; param: any }) {
  const namePath = ["plugins", pluginName, "config", param.key];

  if (param.type === "bool") {
    return (
      <Form.Item label={param.title} name={namePath} valuePropName="checked" tooltip={param.description}>
        <Switch defaultChecked={param.default ?? false} />
      </Form.Item>
    );
  }

  if (param.type === "int" || param.type === "float") {
    return (
      <Form.Item label={param.title} name={namePath} tooltip={param.description}>
        <InputNumber
          style={{ width: "100%" }}
          min={param.min}
          max={param.max}
          defaultValue={param.default}
          step={param.type === "int" ? 1 : 0.1}
        />
      </Form.Item>
    );
  }

  if (param.type === "enum") {
    return (
      <Form.Item label={param.title} name={namePath} tooltip={param.description}>
        <Select options={(param.enum || []).map((x: any) => ({ label: String(x), value: x }))} />
      </Form.Item>
    );
  }

  return (
    <Form.Item label={param.title} name={namePath} tooltip={param.description}>
      <Input defaultValue={param.default ?? ""} />
    </Form.Item>
  );
}

function yamlToFormValues(yamlText: string, plugins: PluginMeta[]) {
  const obj: any = yaml.load(yamlText);
  const values: any = {};

  values.tracking_uri = obj?.mlflow?.tracking_uri ?? "http://localhost:5000";
  values.output_dir = obj?.output?.output_dir ?? "./artifacts";

  const tf = obj?.timeframe ?? {};
  if (tf.last_n_hours != null) {
    values.timeframe_mode = "hours";
    values.last_n_hours = tf.last_n_hours;
  } else if (tf.last_n_traces != null) {
    values.timeframe_mode = "traces";
    values.last_n_traces = tf.last_n_traces;
  } else {
    values.timeframe_mode = "days";
    values.last_n_days = tf.last_n_days ?? 7;
  }

  values.plugins = {};
  const cfgPlugins: any[] = obj?.plugins ?? [];
  const cfgByName: Record<string, any> = {};
  for (const p of cfgPlugins) cfgByName[p.name] = p;

  for (const meta of plugins) {
    const p = cfgByName[meta.name];
    values.plugins[meta.name] = {
      enabled: !!p?.enabled,
      config: { ...(p?.config ?? {}) }
    };
  }

  return values;
}

const DEFAULT_YAML = `mlflow:
  tracking_uri: "http://localhost:5000"

timeframe:
  last_n_days: 7

output:
  output_dir: "./artifacts"
  overwrite: true

schema_bindings: {}

plugins:
  - name: "faq_detector"
    enabled: true
    config:
      top_k_clusters: 20
      min_cluster_size: 15
      min_samples: 5
      include_suggestions: true
      include_answer_stability: true
`;