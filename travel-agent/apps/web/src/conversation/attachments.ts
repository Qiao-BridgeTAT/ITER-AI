export type FileAttachment = {
  id: string;
  kind: "file";
  name: string;
  mimeType: string;
  size: number;
  url: string;
  textPreview?: string;
};

export type LocationAttachment = {
  id: string;
  kind: "location";
  name: string;
  detail: string;
  coordinates?: {
    latitude: number;
    longitude: number;
  };
};

export type ConversationAttachment = FileAttachment | LocationAttachment;

export function formatFileSize(size: number) {
  if (size < 1024) {
    return `${size} B`;
  }

  if (size < 1024 * 1024) {
    return `${Math.round(size / 1024)} KB`;
  }

  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}
