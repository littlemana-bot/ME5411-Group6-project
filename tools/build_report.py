"""Build the Group 6 factual report draft and editable algorithm diagrams.

Requires reportlab and Pillow. The completed analysis and reasoning must be
supplied and reviewed by the project group before submission.
"""
from pathlib import Path
import json
import html

from PIL import Image, ImageDraw, ImageFont
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import A4

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/report'
FIG = ROOT / 'docs/figures'
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def diagram(name, title, labels):
    """Write SVG source and a high-resolution PNG with identical geometry."""
    width, box_h, gap, top = 1500, 95, 36, 100
    height = top + len(labels) * (box_h + gap)
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 28)
    bold = ImageFont.truetype(r'C:\Windows\Fonts\arialbd.ttf', 34)
    draw.text((60, 25), title, font=bold, fill='black')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="60" y="55" font-family="Arial" font-size="34" font-weight="bold">{html.escape(title)}</text>']
    for i, label in enumerate(labels):
        y = top + i * (box_h + gap)
        draw.rectangle((70, y, width-70, y+box_h), fill='#f3f4f5', outline='#555555', width=2)
        svg.append(f'<rect x="70" y="{y}" width="{width-140}" height="{box_h}" fill="#f3f4f5" stroke="#555" stroke-width="2"/>')
        for j, line in enumerate(label.split('\n')):
            ty = y + 17 + j * 34
            draw.text((100, ty), line, fill='black', font=font)
            svg.append(f'<text x="100" y="{ty+27}" font-family="Arial" font-size="28">{html.escape(line)}</text>')
        if i < len(labels)-1:
            cx, y0, y1 = width//2, y+box_h, y+box_h+gap
            draw.line((cx,y0,cx,y1-7),fill='black',width=3)
            draw.polygon([(cx,y1),(cx-9,y1-12),(cx+9,y1-12)],fill='black')
            svg.append(f'<path d="M {cx} {y0} V {y1-8}" stroke="black" stroke-width="3"/><path d="M {cx} {y1} l -9 -12 h 18 z"/>')
    svg.append('</svg>')
    image.save(FIG / f'{name}.png')
    (FIG / f'{name}.svg').write_text('\n'.join(svg), encoding='utf-8')


def main():
    cnn = read('output/task7/cnn/metrics.json')
    svm = read('output/task7/baseline/metrics.json')
    audit = read('output/task7/verification/data_audit.json')
    verified = read('output/task7/verification/reloaded_models.json')
    recognition = read('output/task7/recognition/recognition.json')
    sensitivity = read('output/task7/verification/segmentation_study.json')
    config = read('output/task7/cnn/config.json')
    members_path = ROOT/'docs/members.json'
    members = read('docs/members.json') if members_path.exists() else []
    diagram('image_processing_flow', 'Image processing and character segmentation', [
        'Read RGB chip image and convert to grayscale',
        'Save contrast stretch and mean filter comparisons\nDemonstration branch only',
        'Crop raw grayscale ROI  x 10 to 980  y 185 to 350',
        'Gaussian smoothing sigma 3 minus background sigma 25',
        'Threshold response above 10\nKeep components with area at least 100 and height at least 66',
        'Estimate character width and split wide components\nUse local minima of vertical foreground projection',
        'Sort boxes from left to right and export ten crops\nSave grayscale binary and 64 by 64 normalized versions'])
    diagram('cnn_architecture', 'CNN architecture for 32 by 32 inputs', [
        'Input  1 channel by 32 by 32  ink values in 0 to 1',
        'Convolution 3 by 3  16 channels  padding 1  ReLU\nMax pool 2 by 2  output 16 by 16 by 16',
        'Convolution 3 by 3  32 channels  padding 1  ReLU\nMax pool 2 by 2  output 32 by 8 by 8',
        'Flatten 2048 features\nDense 128  ReLU  dropout probability 0.3 during training',
        'Dense 7  softmax\nClasses  0  4  7  8  A  D  H'])
    diagram('classical_flow', 'PCA and SVM classification pipeline', [
        'Crop ink bounding box with padding\nPad to square and resize to 32 by 32',
        'Flatten into 1024 pixel features',
        'StandardScaler  fitted on training data only',
        'PCA 100 components  fitted on training data only',
        'RBF SVM  C equals 10  gamma equals scale',
        'Select preprocessing and parameters on validation data\nEvaluate selected pipeline on held out test data'])
    diagram('evaluation_flow', 'Data isolation and evaluation protocol', [
        'Read 7112 labeled images\nHash decoded pixels and group exact duplicates',
        'Allocate whole groups within each class\nTrain 660  validation 102  test 254 images per class',
        'Train candidate models using training data only\nSelect settings using validation data only',
        'Fit selected pipelines and evaluate the test images\nNo chip image is used for training or model selection',
        'Reload saved models and recompute test predictions\nVerify arrays confusion matrices and reports',
        'Evaluate ten chip characters and nine fixed segmentation settings\nSave results environment timing records and source hashes'])

    # Rebuild the contact sheet with enough width for every preprocessing label.
    import sys
    import numpy as np
    sys.path.insert(0, str(ROOT))
    from task7.data import preprocess, CLASSES
    split = read('output/task7/split.json')
    picks = split['train'][:8]
    paths = [p for c in CLASSES for p in sorted((ROOT/'dataset_2026_updated/dataset_2026'/c).glob('*.png'))]
    sheet = Image.new('RGB', (1600, 850), 'white')
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf', 25)
    draw.text((15, 15), 'Preprocessing at 32 by 32  none  crop  crop_pad', fill='black', font=font)
    for row, mode in enumerate(('none', 'crop', 'crop_pad')):
        for col, index in enumerate(picks):
            raw = np.asarray(Image.open(paths[index]).convert('L'))
            tile = Image.fromarray(np.uint8(preprocess(raw, mode=mode)*255)).resize((180,180))
            x,y = col*200+10,60+row*260
            sheet.paste(tile.convert('RGB'), (x,y))
            draw.text((x,y+190), f'{mode} {paths[index].parent.name}', fill='black', font=font)
    sheet.save(FIG/'classifier_preprocessing.png')

    pdfmetrics.registerFont(TTFont('Arial', r'C:\Windows\Fonts\arial.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Bold', r'C:\Windows\Fonts\arialbd.ttf'))
    pdfmetrics.registerFont(TTFont('Arial-Italic', r'C:\Windows\Fonts\ariali.ttf'))
    pdfmetrics.registerFontFamily('Arial',normal='Arial',bold='Arial-Bold',italic='Arial-Italic',boldItalic='Arial-Bold')
    styles = {
        'body': ParagraphStyle('Body', fontName='Arial',fontSize=11,leading=12.7,spaceAfter=8),
        'heading': ParagraphStyle('Heading',fontName='Arial-Bold',fontSize=15,leading=18,spaceAfter=15),
        'sub': ParagraphStyle('Sub',fontName='Arial-Bold',fontSize=11,leading=12.7,spaceBefore=9,spaceAfter=7),
        'caption': ParagraphStyle('Caption',fontName='Arial',fontSize=11,leading=12.7,spaceBefore=6,spaceAfter=12),
        'cell': ParagraphStyle('Cell',fontName='Arial',fontSize=11,leading=12.7),
        'title': ParagraphStyle('Title',fontName='Arial-Bold',fontSize=23,leading=28,spaceAfter=18),
    }
    story, md, section_titles = [], [], []
    width = A4[0]-108
    def p(text, kind='body'):
        story.append(Paragraph(html.escape(str(text)),styles[kind]))
        md.append(str(text)+'\n')
    def page(title):
        if story:story.append(PageBreak())
        story.append(Paragraph(html.escape(title),styles['heading']))
        md.append('\n## '+title+'\n')
        section_titles.append(title)
    def table(headers, rows, widths=None):
        values=[headers]+[list(map(str,r)) for r in rows]
        paragraphs=[[Paragraph(html.escape(str(v)),styles['cell']) for v in row] for row in values]
        tbl=Table(paragraphs,colWidths=widths or [width/len(headers)]*len(headers),repeatRows=1,hAlign='LEFT')
        tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E8EBEF')),
            ('GRID',(0,0),(-1,-1),0.5,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
        story.extend([tbl,Spacer(1,12)])
        md.extend(['| '+' | '.join(map(str,headers))+' |','| '+' | '.join(['---']*len(headers))+' |'])
        md.extend('| '+' | '.join(map(str,r))+' |' for r in rows)
        md.append('')
    def fig(path, caption, max_h=300):
        fp=ROOT/path
        im=Image.open(fp); scale=min(width/im.width,max_h/im.height)
        story.append(PDFImage(str(fp),width=im.width*scale,height=im.height*scale,hAlign='CENTER'))
        p(caption,'caption')
        # The source is in docs, so all paths are relative to that directory.
        md.append(f'![{caption}]({Path("..").joinpath(path).as_posix()})\n')
    def sub(text):p(text,'sub')
    def evidence(text):p('Evidence: '+text)

    # Deliberate pages pair related figures and tabular evidence without padding to a minimum length.
    story.append(Spacer(1,70))
    p('Microchip Character Recognition','title')
    p('ME5411 Robot Vision and AI')
    p('Group 6 Technical Report Draft')
    p('Experiment records dated 14 October 2026')
    story.append(Spacer(1,20))
    table(['Member','Name','Student number'],[(str(i+1), m.get('name','TO BE PROVIDED'),m.get('student_number','TO BE PROVIDED')) for i,m in enumerate(members or [{}]*4)])
    p('Draft for group review. Member details, course-method eligibility and group-authored discussion are not yet confirmed. This document must not be represented as an approved final submission.')
    p('The assignment permits language editing with LLMs but requires ideas, analysis and reasoning to originate from the project group. This draft organizes existing implementation records, figures and measured results. Section 25 identifies the passages that the group must supply and approve.')
    p('Submission deadline 20 November 2026 at 23:59 Singapore time')

    page('1 Scope and recorded outcomes')
    p('This draft documents the software pipeline implemented for Tasks 1 to 7 of the ME5411 group project [1]. The fixed chip image is processed into ten character crops. A convolutional neural network implemented with NumPy and a non-CNN pipeline using standardization, PCA and an RBF SVM classify the seven character categories.')
    p('The current test split contains 1,778 images. Exact decoded duplicates are grouped before allocation to avoid identical images crossing training, validation and test sets. The CNN makes 14 errors and PCA plus SVM makes 16 errors. On the normalized chip crops, the recorded strings are HD44780A00 and H044780A00 respectively.')
    table(['Record','CNN','PCA plus SVM'],[
        ('Test accuracy',f"{cnn['report']['accuracy']:.4f}",f"{svm['report']['accuracy']:.4f}"),
        ('Macro F1',f"{cnn['report']['macro']['f1']:.4f}",f"{svm['report']['macro']['f1']:.4f}"),
        ('Test errors','14 of 1778','16 of 1778'),('Normalized chip crops','10 of 10','9 of 10')],[width*.44,width*.28,width*.28])
    p('These are observations from one fixed split and one chip photograph. They do not establish general superiority, statistical significance or performance on unseen chip photographs. Explanations and lessons learned must be supplied by the group.')
    sub('Report navigation')
    p('Sections 2 to 4 cover data and evaluation. Sections 5 to 10 document Tasks 1 to 6. Sections 11 to 17 document both classifiers. Sections 18 to 22 present comparison, chip recognition and verification. Sections 23 to 26 cover reproduction, group completion and references.')

    page('2 Input images and labels')
    p('The supplied chip file used locally is Test_Image_2026.bmp, a 990 by 367 RGB image. The task statement names charact2.bmp. The group should confirm that the locally supplied updated image is the intended course version. The required target line is HD44780A00, comprising ten positions and seven distinct categories.')
    fig('output/segmentation/01_original.png','Figure 1 Original chip photograph used for image processing',230)
    p('The classification dataset has seven directories named 0, 4, 7, 8, A, D and H. Each directory contains 1,016 PNG images, giving 7,112 images in total. The classifier inputs are 128 by 128 grayscale images before preprocessing. The category order is fixed consistently in task7/data.py.')
    fig('output/task7/cnn/dataset_samples.png','Figure 2 Sample character images in category order',240)

    page('3 Duplicate audit and dataset allocation')
    p('Each image is decoded to grayscale and its pixel bytes are hashed with SHA-256. Groups with identical hashes are assigned wholly to one partition. The audit found 651 exact-duplicate groups and no conflicting labels. The earlier image-wise split had 370 groups crossing partitions; 253 groups appeared in both training and test. Those earlier results are retained in history but are not the current evaluation.')
    p('Within each category, groups are shuffled with seed 42. A subset-sum allocation selects whole groups to meet exact test and validation image counts; remaining groups form training data. The implementation raises an error if exact allocation without breaking groups is impossible.')
    table(['Partition','Per category','Total','Purpose'],[('Training','660','4620','Model fitting'),('Validation','102','714','Configuration selection'),('Test','254','1778','Final evaluation')],[width*.25,width*.18,width*.16,width*.41])
    p('Training and validation jointly account for 75 percent of the supplied dataset; the test partition accounts for 25 percent. Only the training subset is used to fit the current final models. Validation remains separate for model selection and checkpoint selection.')
    p('The current audit confirms complete coverage, mutually exclusive image indices, reproducibility from the grouped seed and zero exact duplicate groups crossing partitions. Pixel hashes do not detect near-duplicates or related typefaces. No author or font grouping metadata is available in the supplied directory structure.')
    evidence('task7/data.py; task7/audit.py; output/task7/verification/data_audit.json')

    page('4 Evaluation protocol')
    fig('docs/figures/evaluation_flow.png','Figure 3 Data isolation and evaluation workflow',460)
    p('Candidate settings are ranked by validation accuracy. The test set is not used to choose preprocessing, CNN architecture, learning rate, PCA dimension or SVM kernel settings. The chip image is not used for classifier fitting or classifier selection. Test predictions are recomputed after model loading solely to check reproducibility.')

    page('5 Image processing workflow')
    fig('docs/figures/image_processing_flow.png','Figure 4 Image processing and segmentation workflow',470)
    p('Contrast enhancement and mean filtering are saved as demonstration outputs. The actual segmentation branch uses the raw grayscale ROI, Gaussian smoothing and local background subtraction. The demonstration images are not chained into the segmentation input.')

    page('6 Task 1 Contrast enhancement')
    p('The RGB image is converted to grayscale with Pillow. ImageOps.autocontrast produces the saved enhancement demonstration [3]. The operation rescales the image intensity range; it does not learn parameters from classification data. Both the original and enhanced images are preserved for comparison.')
    fig('output/segmentation/02_contrast.png','Figure 5 Grayscale image after automatic contrast stretching',245)
    p('The main segmentation method uses a separate local-response operation, documented in Section 8. The code therefore exposes both a global contrast demonstration and a locally background-corrected response. The group should describe the visual changes it observes and the reasons for retaining the selected branch.')
    sub('Group observation to complete')
    p('TO BE COMPLETED BY THE GROUP: describe whether character strokes and background texture become clearer or more prominent in Figure 5, and explain how this affects the later tasks. Do not infer quantitative enhancement accuracy without a reference image.')
    evidence('segment_characters.py; output/segmentation/02_contrast.png')

    page('7 Task 2 Mean filtering')
    p('A normalized square averaging filter is implemented explicitly with NumPy. For a window of side k, the output is the sum of k squared neighborhood intensities divided by k squared. Image borders are replicated, and the resulting intensities are rounded to uint8. The required 5 by 5 filter is evaluated alongside 3 by 3 and 7 by 7 filters.')
    for k in (3,5,7):
        fig(f'output/segmentation/03_mean_{k}x{k}.png',f'Figure {5+(k-1)//2} Mean filter with a {k} by {k} neighborhood',125)
    p('The mean-filter images are saved for the assignment comparison. They are distinct from the Gaussian-filter branch used for character extraction. The implementation has been checked to preserve a constant-valued image, including the replicated borders.')
    p('TO BE COMPLETED BY THE GROUP: compare background smoothing, thin-stroke visibility and neighboring-character separation across these three images, using specific observed locations.')

    page('8 Tasks 3 and 4 Crop and binarization')
    p('The fixed ROI has coordinates (10, 185, 980, 350), with right and bottom boundaries excluded. It is 970 by 165 pixels and contains the required character line. This is a manually selected ROI for the supplied fixed image, rather than a general text-line detector.')
    fig('output/segmentation/04_roi.png','Figure 9 Grayscale subimage containing the required character line',105)
    p('Let S be the Gaussian-smoothed crop with radius 3 and B the background estimate with radius 25. The local response is R = S - B. Foreground is defined as R greater than 10. Pillow approximates Gaussian filtering with extended box filters; its radius parameter represents the Gaussian standard deviation [4]. The response visualization adds 128 before clipping, whereas the binary threshold is applied to the unshifted floating-point response.')
    fig('output/segmentation/07_local_contrast.png','Figure 10 Local response visualization with an intensity offset of 128',105)
    fig('output/segmentation/08_binary.png','Figure 11 Binary foreground obtained from the local response',105)
    evidence('segment_characters.py; segmentation.json')

    page('9 Tasks 5 and 6 Components and boundaries')
    p('Four-connected foreground components are found using an explicit stack traversal. A component is retained when its area is at least 100 foreground pixels and its height is at least 40 percent of the ROI height, equivalent to 66 pixels. The filter does not use known character labels.')
    fig('output/segmentation/09_cleaned.png','Figure 12 Binary image after component area and height filtering',140)
    p('Boundary pixels are computed as foreground pixels that are not interior under a four-neighbor test. This produces both outside boundaries and boundaries surrounding character holes. The output is a boundary-pixel image; it does not encode ordered contour chains or contour hierarchy.')
    fig('output/segmentation/10_contours.png','Figure 13 Foreground boundary pixels including character holes',140)
    p('Retained connected components can contain touching characters. A later projection-based step subdivides wide components, so the number of retained components need not equal the number of exported character images.')
    evidence('components and split_boxes in segment_characters.py')

    page('10 Touching characters and crop export')
    p('The median width of the narrower half of candidate components estimates a typical character width. A component wider than 1.45 times this estimate is treated as a possible multi-character region. The number of pieces is estimated from its width. A five-column averaged vertical foreground projection locates weak columns near the expected separation positions. No target string or fixed character count is used in this splitting step.')
    fig('output/segmentation/13_projection.png','Figure 14 Vertical foreground projection and cuts within touching components',130)
    fig('output/segmentation/11_annotated.png','Figure 15 ROI and ten exported bounding boxes on the original image',185)
    fig('output/segmentation/12_character_sheet.png','Figure 16 Exported characters in left to right order',90)
    p('For each position, the pipeline saves a grayscale crop, a white-on-black binary crop, and a 64 by 64 normalized black-on-white crop. Normalization preserves aspect ratio, centers the image and retains a six-pixel white margin. Antialiased resizing introduces grayscale edge values. Box coordinates and foreground counts are stored in segmentation.json.')

    page('11 Classifier preprocessing')
    p('Both classifiers use the same category definitions and dataset partitions. Three preprocessing modes are evaluated: whole-frame resizing, ink bounding-box cropping followed by resizing, and cropping followed by square padding and resizing. The ink threshold is 128 and the crop is expanded by two pixels where possible. The resulting image tensor has ink near 1 and background near 0.')
    fig('docs/figures/classifier_preprocessing.png','Figure 17 Preprocessing examples generated from the training partition',245)
    table(['Mode','Geometry'],[('none','Resize full image'),('crop','Crop and resize with possible aspect change'),('crop_pad','Crop pad to square and resize')],[width*.25,width*.75])
    p('The preprocessing transformation itself uses fixed operations, not fitted dataset statistics. Fitted standardization and PCA belong to the classical classifier and are restricted to training data. Cached tensors can be regenerated and are not included in the submission package.')

    page('12 CNN architecture and implementation')
    fig('docs/figures/cnn_architecture.png','Figure 18 Selected CNN architecture',375)
    p('The network implements convolution through im2col and matrix multiplication, 2 by 2 max pooling, fully connected layers, ReLU, inverted dropout and softmax cross entropy. Forward and backward operations are written in NumPy [2]. The selected architecture has 267,975 trainable scalar parameters.')
    p('The model starts from seeded initialization. No pretrained OCR model or external recognition service is used. The seven-way output is mapped to the fixed category order 0, 4, 7, 8, A, D, H. Dropout is active during fitting and disabled during validation and prediction.')
    evidence('task7/cnn.py; output/task7/cnn/config.json')

    page('13 CNN fitting and numerical checks')
    p('Mini-batches contain 64 images. The final selected learning rate is 0.003, dropout probability is 0.3 and weight-decay coefficient is 0.0001. Adam uses beta1 = 0.9, beta2 = 0.999 and epsilon = 1e-8 [8]. The implemented penalty is added to weight gradients before Adam moment updates; it is not a separate AdamW update. Reported cross entropy excludes that penalty term.')
    p('Each candidate is trained for eight epochs. The final selected model is trained for 25 epochs. Validation accuracy is computed each epoch and the best state is copied. Training does not stop early; after all epochs the best validation state is restored. In the current run, the best epoch is 5.')
    table(['Setting','Value'],[('Split seed','42'),('Network and fitting seed','0'),('Final epochs','25'),('Batch size','64'),('Learning rate','0.003'),('Weight decay','0.0001'),('Dropout probability','0.3')],[width*.56,width*.44])
    p('A centered finite-difference check samples six entries per parameter tensor in a small network using float64. Eight tensors are checked. The largest recorded relative error is 1.35e-6, below the configured failure threshold of 1e-4. This is a sampled numerical check, not a proof of every derivative or a complete network test.')

    page('14 CNN preprocessing and hyperparameter results')
    p('The seven candidate settings change preprocessing, input size, learning rate, channel widths or augmentation. Where enabled, augmentation uses translation from a padded crop and contrast or brightness jitter. Selection uses the maximum validation accuracy reached within the eight fitting epochs.')
    table(['Configuration','Validation accuracy','Best epoch','Seconds'],[(e['name'],f"{e['val_accuracy']:.4f}",e['best_epoch'],f"{e['train_seconds']:.2f}") for e in cnn['experiments']],[width*.4,width*.25,width*.16,width*.19])
    p('The recorded winner is croppad32_na: cropped and square-padded 32 by 32 inputs, channels 16 and 32, learning rate 0.003 and no random image augmentation. It reaches validation accuracy 0.9972. All configurations still use dropout during training. The paired croppad32 setting differs by using image augmentation and records validation accuracy 0.9958.')
    p('The 24 by 24 setting records 0.9902 validation accuracy. The smaller channel setting records 0.9916 and has 133,351 parameters. These values document this single-seed study. The group should explain the tradeoffs it considers justified and avoid interpreting one run as a general rule about augmentation or architecture size.')
    evidence('output/task7/cnn/experiments.json')

    page('15 CNN training history and test errors')
    fig('output/task7/cnn/training_curves.png','Figure 19 Final CNN fitting history across 25 epochs',330)
    p('The restored checkpoint makes 14 errors on 1,778 test images, giving accuracy 0.9921 and macro F1 0.9921. The most frequent recorded confusions are H as A in six cases and A as H in three cases. Training accuracy in the history is computed on each mini-batch after its update, not as a separate complete training-set pass.')
    fig('output/task7/cnn/errors.png','Figure 20 Saved CNN error examples with true and predicted labels',130)

    page('16 PCA and SVM method')
    fig('docs/figures/classical_flow.png','Figure 21 Selected classical classification pipeline',400)
    p('StandardScaler stores training feature means and standard deviations [5]. PCA is fitted to standardized training pixels and keeps 100 components [6]. The selected RBF SVM uses C = 10 and gamma = scale [7]. The RBF kernel compares transformed feature vectors using exp of minus gamma times squared Euclidean distance. Multiclass SVC uses the library implementation. No CNN-derived features are used.')
    p('COURSE ELIGIBILITY TO BE CONFIRMED: no Part 2 lecture notes have been supplied. External API documentation describes the algorithms but cannot prove that they were covered in this course.')

    page('17 Classical preprocessing and PCA experiments')
    p('The study first compares four preprocessing settings at 100 PCA components, C = 10 and gamma = scale. It then compares PCA dimensions for the selected preprocessing. The ordering of candidates resolves equal rounded validation scores by the retained candidate order; the selection is not based on test performance.')
    table(['Preprocessing','Size','Validation accuracy'],[(e['preprocessing'],e['size'],f"{e['val_accuracy']:.4f}") for e in svm['study']['preprocessing']],[width*.46,width*.18,width*.36])
    table(['PCA dimensions','Validation accuracy','Variance retained'],[(e['pca_components'],f"{e['val_accuracy']:.4f}",f"{e['explained_variance']:.4f}") for e in svm['study']['pca']],[width*.3,width*.35,width*.35])
    fig('output/task7/baseline/pca_variance.png','Figure 22 Cumulative explained variance for the selected PCA transform',200)

    page('18 SVM kernel parameter results')
    p('After selecting preprocessing and PCA dimension, C is tested at 1, 10 and 100. Gamma is tested at scale, 0.001 and 0.01. Every fitted scaler and PCA transform in the study uses training data only. Validation labels score the candidates but do not enter their fitting.')
    table(['C','Gamma','Validation accuracy'],[(e['C'],e['gamma'],f"{e['val_accuracy']:.4f}") for e in svm['study']['svm_grid']],[width*.2,width*.35,width*.45])
    p('The retained configuration uses crop_pad at 32 pixels, 100 components, C = 10 and gamma = scale. The selected PCA retains 0.9161 of standardized training variance, and the fitted SVM has 886 support vectors. Final fitting does not merge validation data into training.')
    fig('output/task7/baseline/errors.png','Figure 23 Saved PCA plus SVM error examples',140)
    p('The final classical model makes 16 test errors. The most frequent confusion is A as H in seven cases. Other recorded errors include 0 as D and A as 4, each in two cases.')

    page('19 Effectiveness on the shared test partition')
    fig('output/task7/cnn/confusion_matrix.png','Figure 24 CNN confusion matrix with true categories by row',240)
    fig('output/task7/baseline/confusion_matrix.png','Figure 25 PCA plus SVM confusion matrix on the same test indices',240)
    p('There are 254 test images per category. Both matrices sum to 1,778. The CNN diagonal sums to 1,764 and the classical-model diagonal sums to 1,762. The accuracy difference is two correctly classified images on this partition.')

    page('20 Class metrics and paired predictions')
    table(['Class','Support','CNN F1','PCA plus SVM F1'],[(c,254,f"{cnn['report']['per_class'][str(i)]['f1']:.4f}",f"{svm['report']['per_class'][str(i)]['f1']:.4f}") for i,c in enumerate('0478ADH')],[width*.17,width*.19,width*.27,width*.37])
    p('Accuracy is the number of correct predictions divided by the number of test images. For each category, precision uses predicted positives and recall uses actual positives. F1 is the harmonic mean of precision and recall. Macro F1 averages the category F1 scores. The implementation reports rounded per-class scores and macro values.')
    table(['Paired outcome','Count'],[(k.replace('_',' '),v) for k,v in verified['paired_outcomes'].items()],[width*.7,width*.3])
    p('Both classifiers are correct for 1,752 test images and wrong for four. Twelve images are correctly classified only by the CNN and ten only by PCA plus SVM. No confidence interval or significance claim is reported. Repeated seeds, independent font groups and additional photographs were not part of this study.')

    page('21 Efficiency measurement')
    table(['Measurement','CNN','PCA plus SVM'],[
        ('Final fitting seconds',f"{cnn['train_seconds']:.2f}",f"{svm['train_seconds']:.2f}"),
        ('Single image ms',f"{cnn['timing']['ms_per_image_single']:.3f}",f"{svm['timing']['ms_per_image_single']:.3f}"),
        ('Batch ms per image',f"{cnn['timing']['ms_per_image_batched']:.3f}",f"{svm['timing']['ms_per_image_batched']:.3f}"),
        ('Model bytes',cnn['model_bytes'],svm['model_bytes'])],[width*.48,width*.26,width*.26])
    p('Final fitting excludes image loading, image preprocessing and hyperparameter search for both methods. The classical timer includes scaler fitting, PCA fitting, feature transformations and SVM fitting. The CNN timer includes optimization, validation each epoch and checkpoint selection. These are costs of the implemented fitting workflows rather than identical optimization workloads.')
    fig('output/task7/comparison/latency.png','Figure 26 Warmed inference latency including classical feature transforms',245)
    p('Inference starts from a preprocessed image tensor and ends at a predicted class. Both methods are warmed up and timed three times; the median is reported. Batch size is 256, and 200 single images are selected with seed 42. Classical inference includes scaler and PCA transforms. Neither timing includes file I/O or image resizing. Stored bytes are compared directly; CNN trainable parameters are not treated as equivalent to SVM and PCA stored state.')

    page('22 End to end chip recognition')
    table(['Input','CNN string','Hits','SVM string','Hits'],[(k,r['cnn_string'],f"{r['cnn_correct']}/10",r['pca_svm_string'],f"{r['pca_svm_correct']}/10") for k,r in recognition.items()],[width*.22,width*.28,width*.10,width*.28,width*.12])
    p('The expected line HD44780A00 is used only to score the predictions by position. The segmentation algorithm does not use it to decide split positions or crop count. The normalized input is the cleaned binary crop after padding and resizing. The gray input is the raw grayscale crop with polarity correction. The gray_enhanced input applies autocontrast, Gaussian smoothing of radius 1.5 and an Otsu threshold to estimate the ink bounding box.')
    p('The current normalized predictions are correct at all ten positions for the CNN and at nine positions for PCA plus SVM. Both classifiers are correct at two positions on the raw gray variant. Enhanced gray results are eight and four correct positions. These are measurements of different inputs from one photograph, not an independent multi-image robustness benchmark.')
    p('CNN displayed scores are softmax outputs. The SVM display applies softmax to decision-function scores, which are not calibrated class probabilities. They should not be interpreted as directly comparable confidence estimates.')
    evidence('task7/recognize.py; output/task7/recognition/recognition.json')

    page('23 Segmentation parameter sensitivity')
    p('A fixed grid tests Gaussian smoothing radii 2, 3 and 4 with local thresholds 8, 10 and 12. Background radius remains 25, minimum area remains 100 and the ROI remains fixed. No classifier is retrained or selected using these chip experiments.')
    table(['Radius','Threshold','Count','CNN correct string','SVM correct string'],[(r['blur'],r['threshold'],r['count'],'Yes' if r['cnn_exact_string'] else 'No','Yes' if r['svm_exact_string'] else 'No') for r in sensitivity['settings']],[width*.16,width*.2,width*.16,width*.24,width*.24])
    p('All nine settings produce ten crops, the CNN string HD44780A00 and the SVM string H044780A00. Mean box intersection-over-union relative to the default boxes ranges from 0.960 to 1.000. This is a stability measure relative to the implementation output, not detection accuracy against annotated ground-truth boxes.')
    p('The ROI and splitting assumptions are specific to this supplied image. Width-based splitting assumes roughly comparable character widths; severe broken strokes, variable-width fonts, rotation, new backgrounds and other chip layouts have not been evaluated. The group should state which of these limits matter to its intended scope.')

    page('24 Reproducibility and software environment')
    p('The complete experiment run takes 867.8 seconds in the recorded environment. Model loading and uncached preprocessing reproduce the saved test predictions, confusion matrices and reports exactly. A separate quick run finishes successfully and SHA-256 checks confirm that all 37 pre-existing formal result files remain unchanged.')
    table(['Component','Recorded version'],[('Python','3.14.6'),*audit['environment']['versions'].items()],[width*.48,width*.52])
    p('The recorded platform is Windows 11 on an AMD64 processor. OPENBLAS_NUM_THREADS, OMP_NUM_THREADS and MKL_NUM_THREADS are set to 1. Runtime measurements apply to that environment and should not be transferred directly to other machines. Core package versions are listed in requirements-tested.txt.')
    sub('Run from a new copy')
    for line in ['1  Install Python and create a virtual environment in the extracted project directory.',
                 '2  Run python -m pip install -r requirements.txt.',
                 '3  Extract dataset_2026_updated.zip into dataset_2026_updated so that dataset_2026 is the next directory.',
                 '4  Set the three thread variables to 1 as shown in README.md.',
                 '5  Run python segment_characters.py.',
                 '6  Run python -m task7.audit then python run_task7.py.',
                 '7  Run python -m task7.verify_results and python -m task7.segmentation_study.',
                 '8  Run python -m task7.report_experiments to regenerate the experiment summary.']:
        p(line)
    p('To inspect stored models without fitting, run python -m task7.recognize after segmentation. The --quick option of run_task7.py writes to output/task7_smoke and is not a source of formal metrics. A freshly extracted review package has also passed segmentation, saved-model prediction verification and chip recognition in a new virtual environment with the recorded core package versions. Full model training was not repeated in that second environment. The acceptance record is docs/package_acceptance.json.')

    page('25 Group discussion and final confirmation')
    p('This section requires original group input before submission. The assignment explicitly requires group-originated ideas, analysis and reasoning [1]. No member names, student numbers, contributions, course approvals or original group conclusions have been invented in this draft.')
    sub('Method rationale and lessons learned')
    p('TO BE COMPLETED BY THE GROUP: explain the chosen image-processing branch, the criteria used to judge enhancement and filtering, and the benefits or failures observed when splitting touching characters. Cite specific figures and observed positions.')
    p('TO BE COMPLETED BY THE GROUP: explain the recorded preprocessing and hyperparameter differences, the two classifiers\' confusion patterns, and the cost of the implemented workflows. Separate observations from untested explanations. Include the group\'s lessons learned from the duplicate audit and the raw-photo experiments.')
    sub('Conclusion')
    p('TO BE COMPLETED BY THE GROUP: state the project\'s conclusions within the measured scope, including what was achieved, what remains limited and which follow-up would be justified by the results. Avoid claiming general superiority from two errors or one photograph.')
    sub('Confirmation before final export')
    table(['Required item','Current state'],[('Four names and student numbers','Awaiting group information'),('PCA and SVM covered in Part 2','Awaiting lecture evidence'),('Group discussion and conclusion','Awaiting group text and approval'),('Actual contribution statement','Awaiting member confirmation'),('Draft status removed','Only after preceding items are resolved'),('Canvas upload','Not performed')],[width*.52,width*.48])

    page('26 References and implementation acknowledgments')
    refs=[
        '[1] NUS ME5411 Robot Vision and AI. Group Project AY2627. Supplied assignment statement ME5411_Project_AY2627.pdf and course dataset dataset_2026_updated.zip.',
        '[2] NumPy developers. NumPy documentation. https://numpy.org/doc/stable/',
        '[3] Pillow contributors. ImageOps documentation. https://pillow.readthedocs.io/en/stable/reference/ImageOps.html',
        '[4] Pillow contributors. ImageFilter documentation. https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html',
        '[5] scikit-learn developers. StandardScaler API. https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html',
        '[6] scikit-learn developers. PCA API. https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
        '[7] scikit-learn developers. SVC API. https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html',
        '[8] D. P. Kingma and J. Ba. Adam A Method for Stochastic Optimization. 2014. https://arxiv.org/abs/1412.6980']
    for r in refs:p(r)
    p('Web documentation was checked on 14 October 2026. References [2] to [8] describe the tools and algorithm definitions; project scores and counts come from saved local experiment records, not these references.')
    p('NumPy implements arrays and linear algebra used by the custom CNN and explicit image operations. Pillow provides image decoding, resizing, Gaussian filtering, contrast operations and figure drawing. scikit-learn provides StandardScaler, PCA and SVC; its dependencies include SciPy. Python standard-library tools provide hashing, JSON records and archive handling. ReportLab is used only to assemble this report, not to classify characters.')
    p('This draft and its supporting implementation were prepared with AI assistance. The group must verify that its use of assistance meets the course requirements, acknowledge applicable external resources, and supply its own analysis and reasoning. This statement does not certify compliance on the group\'s behalf.')

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Arial',9)
        canvas.setFillColor(colors.HexColor('#444444'))
        canvas.drawString(54,28,'ME5411 Group 6   Technical draft for group review')
        canvas.drawRightString(A4[0]-54,28,str(doc.page))
        canvas.restoreState()
    doc=SimpleDocTemplate(str(OUT/'ME5411_Group06_Report_Draft.pdf'),pagesize=A4,
                         leftMargin=54,rightMargin=54,topMargin=48,bottomMargin=48,
                         title='Microchip Character Recognition Group 6 Technical Draft',author='Group 6 member information pending')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    (ROOT/'docs/report_draft.md').write_text('# Microchip Character Recognition\n\n'+'\n'.join(md),encoding='utf-8')
    (ROOT/'docs/report_sections.json').write_text(json.dumps(section_titles,indent=2),encoding='utf-8')
    print('Report and editable diagrams generated at',OUT)


if __name__=='__main__':
    main()
