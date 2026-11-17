export function createJobsModelsController({
  elements,
  loadJobHistory,
  loadModelCatalog,
  getDefaultModelId,
  formatLocalTime,
  onCheckJob,
  onDeleteModel,
  onFetchJobResult,
  onRemoveJob,
  onSetDefaultModel,
  onUpdateModelMemo,
  onUseModel,
  setVisible,
  t,
}) {
  const { jobsEmptyEl, jobsTableBodyEl, modelsEmptyEl, modelsTableBodyEl } = elements;

  function createTextCell(doc, text) {
    const td = doc.createElement("td");
    td.textContent = String(text ?? "");
    return td;
  }

  function createButton(doc, labelKey, onClick, { disabled = false } = {}) {
    const button = doc.createElement("button");
    button.type = "button";
    button.className = "secondary";
    button.textContent = t(labelKey);
    button.disabled = !!disabled;
    button.setAttribute("aria-disabled", disabled ? "true" : "false");
    button.addEventListener("click", onClick);
    return button;
  }

  function jobStatusLabel(status, translations) {
    const normalized = String(status || "").trim();
    const key = `jobs.status.${normalized}`;
    const translated = translations?.[key];
    return typeof translated === "string" ? t(key) : normalized || "-";
  }

  function renderModels() {
    const defaultModelId = getDefaultModelId();
    const models = loadModelCatalog();
    modelsTableBodyEl.innerHTML = "";
    setVisible(modelsEmptyEl, models.length === 0);
    if (models.length === 0) return;

    const doc = modelsTableBodyEl.ownerDocument || document;
    for (const model of models) {
      const tr = doc.createElement("tr");
      tr.appendChild(createTextCell(doc, model.model_id || ""));
      tr.appendChild(createTextCell(doc, model.created_at ? formatLocalTime(model.created_at) : "-"));

      const memoTd = doc.createElement("td");
      const memoInput = doc.createElement("input");
      memoInput.type = "text";
      memoInput.value = String(model.memo || "");
      memoInput.placeholder = t("models.table.memo");
      memoInput.addEventListener("change", () => {
        onUpdateModelMemo({ ...model, memo: memoInput.value });
      });
      memoTd.appendChild(memoInput);
      tr.appendChild(memoTd);

      const actionsTd = doc.createElement("td");
      actionsTd.style.whiteSpace = "nowrap";
      actionsTd.appendChild(createButton(doc, "models.action.use", () => onUseModel(model)));
      actionsTd.appendChild(doc.createTextNode(" "));
      actionsTd.appendChild(
        createButton(doc, "models.action.set_default", () => onSetDefaultModel(model.model_id), {
          disabled: defaultModelId === model.model_id,
        }),
      );
      actionsTd.appendChild(doc.createTextNode(" "));
      actionsTd.appendChild(createButton(doc, "models.action.delete", () => onDeleteModel(model.model_id)));
      tr.appendChild(actionsTd);

      if (defaultModelId && defaultModelId === model.model_id) {
        tr.setAttribute("data-default-model", "true");
      }
      modelsTableBodyEl.appendChild(tr);
    }
  }

  function renderJobHistory({ translations } = {}) {
    const jobs = loadJobHistory();
    jobsTableBodyEl.innerHTML = "";
    setVisible(jobsEmptyEl, jobs.length === 0);
    if (jobs.length === 0) return;

    const doc = jobsTableBodyEl.ownerDocument || document;
    for (const job of jobs) {
      const tr = doc.createElement("tr");
      tr.appendChild(createTextCell(doc, job.job_id || ""));

      const statusLines = [jobStatusLabel(job.status, translations)];
      if (job.error_code) statusLines.push(`error_code=${job.error_code}`);
      if (job.request_id) statusLines.push(`request_id=${job.request_id}`);
      if (job.message) statusLines.push(`message=${job.message}`);
      tr.appendChild(createTextCell(doc, statusLines.filter(Boolean).join("\n")));

      const progress = Number(job.progress);
      tr.appendChild(createTextCell(doc, Number.isFinite(progress) ? `${progress}%` : "-"));
      tr.appendChild(createTextCell(doc, job.updated_at ? formatLocalTime(job.updated_at) : "-"));

      const actionsTd = doc.createElement("td");
      actionsTd.style.whiteSpace = "nowrap";
      actionsTd.appendChild(createButton(doc, "action.job_check", () => void onCheckJob(String(job.job_id || ""))));
      actionsTd.appendChild(doc.createTextNode(" "));
      actionsTd.appendChild(createButton(doc, "action.job_result", () => void onFetchJobResult(String(job.job_id || ""))));
      actionsTd.appendChild(doc.createTextNode(" "));
      actionsTd.appendChild(createButton(doc, "action.job_remove", () => onRemoveJob(String(job.job_id || ""))));

      tr.appendChild(actionsTd);
      jobsTableBodyEl.appendChild(tr);
    }
  }

  return {
    renderJobHistory,
    renderModels,
  };
}