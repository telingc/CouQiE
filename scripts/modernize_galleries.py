"""One-time migration of legacy gallery writers to externally linked HTML."""
from project_paths import PROJECT_ROOT as ROOT


def main():
    for name in ("experiment9.py","experiment10.py"):
        path=ROOT/"scripts"/name;text=path.read_text(encoding="utf-8")
        if name=="experiment9.py":
            start=text.index("def save_private_gallery(")
            end=text.index("\ndef run_experiment9(",start)
            replacement='''def save_private_gallery(root,run,records,checkpoint_path,*,kernel_size,experiment_number=9):
    """Reference saved comparisons from a private local HTML gallery."""
    private=Path(root)/'data/nju_testset/predictions'/f'exp{experiment_number}'/run.name
    items=[(r['group']+' / '+r['image'],r['comparison_path']) for r in records if r['status']=='complete']
    return write_gallery_html(private/'predictions.html',f'Experiment {experiment_number}: {kernel_size}x{kernel_size}',items,
        f'Checkpoint: {checkpoint_path}. Background is purple. Photos have no independent evaluation masks.')

'''
            text=text[:start]+replacement+text[end:]
            text=text.replace("palette=['black','red','orange','deepskyblue','limegreen']","palette=['mediumpurple','red','orange','deepskyblue','limegreen']")
            text=text.replace("alpha=(values>0).astype(float)*.5","alpha=np.where(values==0,.30,.5)")
            text=text.replace("for c in range(1,5)","for c in range(5)")
            text=text.replace("            save_overlay(image,mask,output_folder/f'{path.stem}_{name}_overlay.jpg')\n","")
            text=text.replace("run/'data/examples'","run/'examples'")
        marker="    cells=[nbformat.v4.new_markdown_cell("
        if marker in text:
            start=text.index(marker);end=text.index("    font=ImageFont",start)
            text=text[:start]+text[end:]
        lines=[]
        for line in text.splitlines():
            if any(token in line for token in ("cell=nbformat.","cell['attachments']","cells.append(cell)","notebook=nbformat.","nbformat.validate(notebook)","import nbformat","import base64")):
                continue
            if "gallery=private/'kernel_comparison.ipynb'" in line:
                line="    gallery=write_gallery_html(private/'kernel_comparison.html','Kernel comparison',[(Path(p).stem,p) for p in comparison_paths])"
            if "gallery=private/'before_after.ipynb'" in line:
                line="    gallery=write_gallery_html(private/'before_after.html','Before and after',[(Path(p).stem,p) for p in comparison_paths])"
            lines.append(line)
        text="\n".join(lines)+"\n"
        text=text.replace("from project_paths import PROJECT_ROOT","from project_paths import PROJECT_ROOT\nfrom gallery import write_gallery_html")
        compile(text,str(path),"exec");path.write_text(text,encoding="utf-8")
    print("Legacy gallery writers now use linked HTML; full-resolution overlays are on demand.")


if __name__=="__main__":main()
