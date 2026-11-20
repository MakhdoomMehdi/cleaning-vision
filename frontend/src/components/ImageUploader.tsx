import { useId, useState, type ChangeEvent } from "react";
import { ALLOWED_IMAGE_TYPES, validateImageFile } from "../lib/imageFile";
import { useObjectUrl } from "../lib/useObjectUrl";

interface ImageUploaderProps {
  label: string;
  multiple?: boolean;
  disabled?: boolean;
  onFilesChange: (files: File[]) => void;
}

function Preview({ file }: { file: File }) {
  const url = useObjectUrl(file);
  return (
    <figure className="overflow-hidden rounded border border-neutral-200 bg-neutral-100">
      {url && <img src={url} alt={`Preview of ${file.name}`} className="h-28 w-full object-cover" />}
      <figcaption className="truncate px-2 py-1 text-xs text-neutral-600">{file.name}</figcaption>
    </figure>
  );
}

export default function ImageUploader({
  label,
  multiple = false,
  disabled = false,
  onFilesChange,
}: ImageUploaderProps) {
  const inputId = useId();
  const [files, setFiles] = useState<File[]>([]);
  const [errors, setErrors] = useState<string[]>([]);

  function handleChange(event: ChangeEvent<HTMLInputElement>) {
    const selected = Array.from(event.target.files ?? []);
    const accepted: File[] = [];
    const rejected: string[] = [];
    for (const file of selected) {
      const problem = validateImageFile(file);
      if (problem) rejected.push(problem);
      else accepted.push(file);
    }
    setErrors(rejected);
    setFiles(accepted);
    onFilesChange(accepted);
    // Allow re-selecting the same file after a rejection.
    event.target.value = "";
  }

  return (
    <div>
      <label htmlFor={inputId} className="block text-sm font-medium text-neutral-700">
        {label}
      </label>
      <input
        id={inputId}
        type="file"
        accept={ALLOWED_IMAGE_TYPES.join(",")}
        multiple={multiple}
        disabled={disabled}
        onChange={handleChange}
        className="mt-1 block w-full text-sm text-neutral-700 file:mr-3 file:rounded file:border file:border-neutral-300 file:bg-white file:px-3 file:py-1.5 file:text-sm hover:file:bg-neutral-50 disabled:opacity-50"
      />
      <p className="mt-1 text-xs text-neutral-500">JPEG, PNG or WebP.</p>

      {errors.length > 0 && (
        <ul role="alert" className="mt-2 space-y-1 text-sm text-red-700">
          {errors.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      )}

      {files.length > 0 && (
        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
          {files.map((file) => (
            <Preview key={`${file.name}-${file.size}-${file.lastModified}`} file={file} />
          ))}
        </div>
      )}
    </div>
  );
}
