export interface GuiRunSnapshot {
  errorText?: string | null;
  resultOutputPath?: string | null;
  jobProgressVisible?: boolean;
  managedJob?: {
    state?: string | null;
  } | null;
}

export function guiRunHasStarted(snapshot: GuiRunSnapshot | null | undefined): boolean;
export function guiRunHasCompleted(snapshot: GuiRunSnapshot | null | undefined): boolean;
export function guiRunHasFailed(snapshot: GuiRunSnapshot | null | undefined): boolean;