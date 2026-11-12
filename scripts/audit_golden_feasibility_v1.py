"""Exploratory structure grouping / registration only; not a Golden defect detector."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import textwrap

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/"data/derived/gt_629_audit_v1"
SPLIT=ROOT/"data/derived/fixed_split_v1/split_manifest.csv"
DATA=ROOT/"data/derived/yolov8_detection_fixed_v1"
OUT=ROOT/"data/derived/golden_feasibility_v1"
SEED=42


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path,rows):
    with path.open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8")


def load(row,color=False):
    im=cv2.imdecode(np.frombuffer((ROOT/row["source_path"]).read_bytes(),dtype=np.uint8),
                    cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
    assert im is not None
    assert im.shape[:2]==(int(row["height"]),int(row["width"]))
    return im


def resample_vector(values,n):
    return cv2.resize(np.asarray(values,dtype=np.float32).reshape(-1,1),(1,n),
                      interpolation=cv2.INTER_AREA).ravel()


def structural_features(gray):
    h,w=gray.shape
    # Descriptor-only downsampling, never used as a registration input.
    thumb=cv2.resize(gray,(512,max(64,round(512*h/w))),interpolation=cv2.INTER_AREA).astype(np.float32)/255
    smooth=cv2.GaussianBlur(thumb,(5,5),0)
    gx=cv2.Sobel(smooth,cv2.CV_32F,1,0,ksize=3)/8
    gy=cv2.Sobel(smooth,cv2.CV_32F,0,1,ksize=3)/8
    xx,yy,xy=float(np.mean(gx*gx)),float(np.mean(gy*gy)),float(np.mean(gx*gy))
    angle=(0.5*math.atan2(2*xy,xx-yy)+math.pi/2)%math.pi
    anisotropy=math.sqrt((xx-yy)**2+4*xy*xy)/(xx+yy+1e-12)
    profile=np.median(smooth,axis=0)
    spectrum=np.abs(np.fft.rfft((profile-profile.mean())*np.hanning(len(profile))))**2
    spectrum[:2]=0
    spectrum[97:]=0
    frequency=int(np.argmax(spectrum))
    period=w/frequency if frequency else float(w)
    strength=float(spectrum[frequency]/(spectrum.sum()+1e-12))
    std=max(float(smooth.std()),0.01)
    spatial=cv2.resize((smooth-smooth.mean())/std,(8,6),interpolation=cv2.INTER_AREA).ravel()
    y_intensity=resample_vector(smooth.mean(axis=1),16)
    y_intensity=(y_intensity-smooth.mean())/std
    y_horizontal=resample_vector(np.mean(np.abs(gy),axis=1),16)
    y_horizontal=y_horizontal/(np.mean(np.abs(gy))+1e-6)
    y_vertical=resample_vector(np.mean(np.abs(gx),axis=1),16)
    y_vertical=y_vertical/(np.mean(np.abs(gx))+1e-6)
    fft=cv2.resize((spectrum/(spectrum.sum()+1e-12)).astype(np.float32).reshape(1,-1),
                   (32,1),interpolation=cv2.INTER_AREA).ravel()
    statistics=np.array([smooth.mean(),smooth.std(),*np.quantile(smooth,[0.1,0.5,0.9])])
    blocks=[
        np.array([math.log(w),math.log(h),math.log(w/h)]),
        np.array([anisotropy*math.cos(2*angle),anisotropy*math.sin(2*angle)]),
        np.array([math.log(period),frequency/32,strength]),
        np.r_[y_intensity,y_horizontal,y_vertical],
        np.r_[spatial,fft],statistics,
    ]
    scalar={
        "dominant_line_orientation_deg":math.degrees(angle),
        "orientation_anisotropy":anisotropy,"dominant_period_px_estimate":period,
        "dominant_cycles_across_width":frequency,"period_spectral_concentration":strength,
        "mean_gray_0_1":float(smooth.mean()),"std_gray_0_1":float(smooth.std()),
        "mean_abs_gradient_x":float(np.mean(np.abs(gx))),
        "mean_abs_gradient_y":float(np.mean(np.abs(gy))),
    }
    return blocks,scalar


def silhouette(distance,labels):
    unique=np.unique(labels)
    scores=[]
    for i,label in enumerate(labels):
        own=np.flatnonzero(labels==label)
        if len(own)==1:
            scores.append(0)
            continue
        a=float(distance[i,own].sum()/(len(own)-1))
        b=min(float(distance[i,labels==other].mean()) for other in unique if other!=label)
        scores.append((b-a)/max(a,b,1e-12))
    return float(np.mean(scores))


def group_features(matrix):
    # Ward agglomeration, one hierarchy; K chosen by descriptor silhouette, not GT.
    n=len(matrix)
    squared=np.maximum(0,np.sum(matrix*matrix,axis=1)[:,None]+
                       np.sum(matrix*matrix,axis=1)[None,:]-2*matrix@matrix.T)
    distance=np.sqrt(squared)
    active=np.ones(n,dtype=bool)
    size=np.ones(n)
    members={i:[i] for i in range(n)}
    ward=squared.copy()
    np.fill_diagonal(ward,np.inf)
    candidates={}
    for k in range(n,2,-1):
        i,j=np.unravel_index(np.argmin(ward),ward.shape)
        a,b=size[i],size[j]
        others=np.flatnonzero(active)
        dab=ward[i,j]
        vals=((a+size[others])*ward[i,others]+(b+size[others])*ward[j,others]-
              size[others]*dab)/(a+b+size[others])
        ward[i,others]=vals
        ward[others,i]=vals
        size[i]=a+b
        members[i]+=members.pop(j)
        active[j]=False
        ward[j,:]=np.inf
        ward[:,j]=np.inf
        ward[i,i]=np.inf
        count=k-1
        if 2<=count<=20:
            labels=np.empty(n,dtype=int)
            for label,indices in members.items():
                labels[indices]=label
            candidates[count]=(silhouette(distance,labels),labels.copy())
    chosen=max(candidates,key=lambda k:(candidates[k][0],-k))
    return candidates[chosen][1],distance,{str(k):v[0] for k,v in candidates.items()},chosen


def representative_indices(indices,rows,distance,n=12):
    pool=list(indices)
    medoid=min(pool,key=lambda i:(float(distance[i,pool].sum()),int(rows[i]["image_id"])))
    selected=[medoid]
    wishes=[
        min(pool,key=lambda i:int(rows[i]["total_annotations"])),
        max(pool,key=lambda i:int(rows[i]["total_annotations"])),
        min(pool,key=lambda i:int(rows[i]["width"])*int(rows[i]["height"])),
        max(pool,key=lambda i:int(rows[i]["width"])*int(rows[i]["height"])),
    ]
    for i in wishes:
        if i not in selected and len(selected)<n:
            selected.append(i)
    for field in ("wafer","position"):
        for value in sorted({rows[i][field] for i in pool}):
            if len(selected)>=n:
                break
            if not any(rows[i][field]==value for i in selected):
                i=min((i for i in pool if rows[i][field]==value),
                      key=lambda i:(float(distance[i,medoid]),int(rows[i]["image_id"])))
                if i not in selected:
                    selected.append(i)
    while len(selected)<min(n,len(pool)):
        others=[i for i in pool if i not in selected]
        seen={rows[i]["capture_group"] for i in selected}
        i=max(others,key=lambda i:(rows[i]["capture_group"] not in seen,
                                  float(min(distance[i,selected])), -int(rows[i]["image_id"])))
        selected.append(i)
    return selected


def contact_sheet(group,rows):
    width,height=440,322
    canvas=Image.new("RGB",(width*4,height*math.ceil(len(rows)/4)+44),"white")
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype("C:/Windows/Fonts/arial.ttf",14)
    draw.text((12,10),group+" | structure candidates; display thumbnails only",font=font,fill="black")
    for i,r in enumerate(rows):
        x=(i%4)*width;y=(i//4)*height+44
        im=Image.fromarray(cv2.cvtColor(load(r,True),cv2.COLOR_BGR2RGB))
        im.thumbnail((width-12,224),Image.Resampling.LANCZOS)
        canvas.paste(im,(x+(width-im.width)//2,y))
        name="\n".join(textwrap.wrap(r["filename"],width=49))
        label=f"{group} / image {r['image_id']} / {r['width']}x{r['height']}\n{r['wafer'] or 'UNKNOWN'} / {r['position'] or 'UNKNOWN'} / annotations {r['total_annotations']}\n{name}"
        draw.multiline_text((x+6,y+226),label,font=font,fill="black",spacing=1)
    canvas.save(OUT/"contact_sheets"/(group+".png"))


def prepare():
    if OUT.exists():
        raise FileExistsError("Do not overwrite Golden feasibility v1")
    rows=read_csv(AUDIT/"image_manifest.csv")
    assignments={r["image_id"]:r for r in read_csv(SPLIT)}
    assert len(rows)==len(assignments)==629
    protected={p.relative_to(ROOT).as_posix():digest(p) for p in (
        AUDIT/"image_manifest.csv",AUDIT/"annotation_manifest.csv",SPLIT,DATA/"data.yaml")}
    for p in DATA.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png",".txt"):
            protected[p.relative_to(ROOT).as_posix()]=digest(p)
    features=[]
    for idx,row in enumerate(rows):
        a=assignments[row["image_id"]]
        assert row["filename"]==a["filename"]
        row["original_export_split"]=row["split"]
        row["split"]=a["split"]
        row["capture_group"]=a["group_key"]
        row["aspect_ratio"]=int(row["width"])/int(row["height"])
        row["source_path"]=(DATA/a["split"]/"images"/row["filename"]).relative_to(ROOT).as_posix()
        assert digest(ROOT/row["source_path"])==a["sha256"]==row["sha256"]
        row["annotation_classes"]=";".join(n for n in ("o","p","s","sc") if int(row[n+"_count"]))
        blocks,scalar=structural_features(load(row))
        features.append(blocks)
        row.update(scalar)
        if (idx+1)%100==0:
            print(f"STRUCTURE_FEATURES={idx+1}/629",flush=True)
    normalized=[]
    for block in range(len(features[0])):
        values=np.stack([f[block] for f in features])
        center=np.median(values,axis=0)
        spread=np.std(values,axis=0)
        floor=0.02 if block==0 else 0.03
        normalized.append(np.clip((values-center)/np.maximum(spread,floor),-6,6)/math.sqrt(values.shape[1]))
    matrix=np.concatenate(normalized,axis=1)
    labels,distance,scores,k=group_features(matrix)
    groups=sorted((np.flatnonzero(labels==v).tolist() for v in np.unique(labels)),
                  key=lambda ids:(-len(ids),min(int(rows[i]["image_id"]) for i in ids)))
    OUT.mkdir(parents=True)
    (OUT/"contact_sheets").mkdir()
    (OUT/"examples").mkdir()
    group_rows=[]
    review={}
    for index,indices in enumerate(groups,1):
        gid=f"G{index:02d}"
        selected=representative_indices(indices,rows,distance)
        for i in indices:
            rows[i]["candidate_group"]=gid
            rows[i]["descriptor_distance_to_medoid"]=float(distance[i,selected[0]])
        contact_sheet(gid,[rows[i] for i in selected])
        # Up to 6 actual native images for pairwise registration per group.
        tested=selected[:6]
        review[gid]=[rows[i]["image_id"] for i in tested]
        values=[rows[i] for i in indices]
        periods=[float(r["dominant_period_px_estimate"]) for r in values]
        group_rows.append({
            "group_id":gid,"image_count":len(indices),
            "wafer_distribution":json.dumps(dict(Counter(r["wafer"] or "UNKNOWN" for r in values)),sort_keys=True),
            "position_distribution":json.dumps(dict(Counter(r["position"] or "UNKNOWN" for r in values)),sort_keys=True),
            "distinct_resolutions":len({(r["width"],r["height"]) for r in values}),
            "width_min":min(int(r["width"]) for r in values),"width_max":max(int(r["width"]) for r in values),
            "height_min":min(int(r["height"]) for r in values),"height_max":max(int(r["height"]) for r in values),
            "period_px_min":min(periods),"period_px_median":float(np.median(periods)),
            "period_px_max":max(periods),"medoid_image_id":rows[selected[0]]["image_id"],
            "contact_sheet_images":len(selected),"registration_subset_images":len(tested),
            "contact_sheet_path":f"contact_sheets/{gid}.png",
        })
    write_csv(OUT/"image_summary.csv",rows)
    write_csv(OUT/"group_summary.csv",group_rows)
    summary={
        "status":"GROUPED_PENDING_REGISTRATION","total_images":len(rows),"candidate_group_count":k,
        "group_image_counts":{r["group_id"]:r["image_count"] for r in group_rows},
        "grouping_method":"One Ward hierarchy on six equal-normalized structural blocks; choose K by highest mean descriptor silhouette for K=2..20, tie favors smaller K. This is exploratory clustering, not defect-threshold tuning.",
        "silhouette_by_k":scores,"group_selection_range_boundary":k in (2,20),
        "structural_blocks":["native geometry","dominant orientation tensor","periodicity",
                             "intensity / gradient row profiles","coarse spatial / spectrum","gray statistics"],
        "annotation_used_for_grouping":False,
        "representative_selection":"Geometric medoid then annotation-count extremes, area extremes, missing wafer/position values and descriptor diversity; no source defect label as image-type ground truth.",
        "registration_subset_image_ids":review,"source_hashes":protected,"seed":SEED,
        "limitations":["All 629 images, including existing Val/Test, are exploratory exposed data for this Golden feasibility review, not an untouched evaluation.",
                       "Structural descriptor grouping does not prove common acquisition scale or defect-free structure.",
                       "Period estimates can have harmonic ambiguity; they are not calibration.",
                       "No detector, detection thresholds, precision/recall/IoU, split changes or training."],
    }
    assert sum(summary["group_image_counts"].values())==629
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    write_json(OUT/"golden_feasibility_summary.json",summary)
    print("GROUPS="+json.dumps(summary["group_image_counts"]),flush=True)


def gradients(im):
    return (cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)/8,
            cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)/8)


def residual(reference,moving,valid):
    rgx,rgy=gradients(reference)
    mgx,mgy=gradients(moving)
    edge=np.abs(rgx-mgx)+np.abs(rgy-mgy)
    energy=np.abs(rgx)+np.abs(rgy)
    return {
        "mae_0_1":float(np.mean(np.abs(reference-moving)[valid])),
        "edge_mae_0_1":float(edge[valid].mean()),
        "normalized_edge_residual":float(edge[valid].mean()/(energy[valid].mean()+1e-8)),
    }


def warp(moving,forward):
    h,w=moving.shape
    result=cv2.warpAffine(moving,forward,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)
    mask=cv2.warpAffine(np.ones_like(moving),forward,(w,h),flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,borderValue=0)>0.999
    mask=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))>0
    return result,mask


def translation(reference,moving):
    # Common native-resolution crops; no resize. Bilinear subpixel translation is explicit.
    a=cv2.GaussianBlur(reference,(0,0),2)
    b=cv2.GaussianBlur(moving,(0,0),2)
    window=cv2.createHanningWindow((a.shape[1],a.shape[0]),cv2.CV_32F)
    shift,response=cv2.phaseCorrelate(a,b,window)
    inverse=np.array([[1,0,shift[0]],[0,1,shift[1]]],dtype=np.float32)
    status="phase_only"
    error=""
    ecc=None
    try:
        ecc,inverse=cv2.findTransformECC(a,b,inverse.copy(),cv2.MOTION_TRANSLATION,
            (cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,40,1e-4),None,5)
        status="ecc_converged"
    except cv2.error as exc:
        error=str(exc).splitlines()[-1]
    forward=cv2.invertAffineTransform(inverse)
    aligned,valid=warp(moving,forward)
    usable=bool(np.isfinite(forward).all() and valid.mean()>=0.70)
    return aligned,valid,forward,{"translation_estimator_status":status,
        "translation_phase_response":float(response),"translation_ecc":float(ecc) if ecc is not None else "",
        "translation_error":error,"translation_support_fraction":float(valid.mean()),
        "translation_usable_support":usable}


def similarity(reference,moving):
    # One fixed ORB + partial-affine estimation; no shear, perspective or dense warp.
    orb=cv2.ORB_create(nfeatures=2000)
    kp1,des1=orb.detectAndCompute(np.uint8(np.clip(moving*255,0,255)),None)
    kp2,des2=orb.detectAndCompute(np.uint8(np.clip(reference*255,0,255)),None)
    detail={"similarity_status":"unavailable","similarity_matches":0,
            "similarity_inliers":0,"similarity_scale":"","similarity_rotation_deg":"",
            "similarity_support_fraction":"","similarity_error":""}
    if des1 is None or des2 is None:
        detail["similarity_error"]="No ORB descriptors"
        return None,None,None,detail
    candidates=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(des1,des2,k=2)
    matches=[a for pair in candidates if len(pair)==2 for a,b in [pair] if a.distance<0.75*b.distance]
    detail["similarity_matches"]=len(matches)
    if len(matches)<6:
        detail["similarity_error"]="Fewer than six ratio-test correspondences"
        return None,None,None,detail
    p=np.float32([kp1[m.queryIdx].pt for m in matches])
    q=np.float32([kp2[m.trainIdx].pt for m in matches])
    cv2.setRNGSeed(SEED)
    forward,inliers=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3,
                                             maxIters=2000,confidence=0.99,refineIters=10)
    if forward is None or not np.isfinite(forward).all():
        detail["similarity_error"]="Partial-affine estimation failed"
        return None,None,None,detail
    scale=float(math.hypot(forward[0,0],forward[1,0]))
    angle=float(math.degrees(math.atan2(forward[1,0],forward[0,0])))
    detail.update(similarity_scale=scale,similarity_rotation_deg=angle,
                  similarity_inliers=int(inliers.sum()))
    if not (0.95<=scale<=1.05 and abs(angle)<=3):
        detail["similarity_status"]="rejected_outside_small_transform"
        detail["similarity_error"]="Fixed feasibility bounds: scale 0.95..1.05, rotation +/-3deg"
        return None,None,None,detail
    aligned,valid=warp(moving,forward.astype(np.float32))
    detail["similarity_support_fraction"]=float(valid.mean())
    if valid.mean()<0.70:
        detail["similarity_status"]="rejected_insufficient_common_support"
        return None,None,None,detail
    detail["similarity_status"]="estimated_within_bounds"
    return aligned,valid,forward,detail


def save_gray(path,im):
    Image.fromarray(np.uint8(np.clip(np.rint(im*255),0,255))).save(path)


def annotations_for_query(row,annotations,crop,forward):
    polygons=[]
    for a in annotations[row["image_id"]]:
        x,y,w,h=(float(a[k]) for k in ("bbox_x","bbox_y","bbox_width","bbox_height"))
        points=np.float32([[x-crop[0],y-crop[1]],[x+w-crop[0],y-crop[1]],
                           [x+w-crop[0],y+h-crop[1]],[x-crop[0],y+h-crop[1]]])
        transformed=cv2.transform(points.reshape(1,-1,2),np.asarray(forward,dtype=np.float32))[0]
        polygons.append((a["category_name"],transformed))
    return polygons


def example(gid,i,rows,aligned,masks,forward,crops,annotations,base,records,kind):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    query=rows[i]
    others=[np.where(m,im,np.nan) for j,(im,m) in enumerate(zip(aligned,masks)) if j!=i]
    assert len(others)>=2
    pseudo=np.nanmedian(np.stack(others),axis=0)
    valid=masks[i].copy()
    for j,m in enumerate(masks):
        if j!=i:
            valid &= m
    assert valid.any()
    diff=np.abs(aligned[i]-pseudo)
    polygons=annotations_for_query(query,annotations,crops[i],forward[i])
    gt=np.zeros_like(valid,dtype=np.uint8)
    for _,p in polygons:
        cv2.fillConvexPoly(gt,np.rint(p).astype(np.int32),1)
    gt=(gt>0)&valid
    outside=(~gt)&valid
    gx,gy=gradients(pseudo)
    edge=np.abs(gx)+np.abs(gy)
    values={
        "example_kind":kind,"query_image_id":query["image_id"],
        "query_annotation_count":int(query["total_annotations"]),
        "query_excluded_from_golden":True,
        "loo_common_support_fraction":float(valid.mean()),
        "loo_mae_0_1":float(diff[valid].mean()),
        "loo_bbox_inside_mae_0_1":float(diff[gt].mean()) if gt.any() else None,
        "loo_bbox_outside_mae_0_1":float(diff[outside].mean()) if outside.any() else None,
        "loo_bbox_outside_gradient_weighted_absdiff_0_1":float(
            np.sum(diff[outside]*edge[outside])/(np.sum(edge[outside])+1e-12)) if outside.any() else None,
        "unannotated_pixels_not_automatically_clean":True,
    }
    folder=base/kind
    folder.mkdir()
    raw=load(query).astype(np.float32)/255
    x,y,w,h=crops[i]
    original=raw[y:y+h,x:x+w]
    save_gray(folder/"representative_original.png",original)
    save_gray(folder/"aligned_image.png",np.where(valid,aligned[i],0))
    save_gray(folder/"pseudo_golden_leave_one_out.png",np.where(valid,pseudo,0))
    save_gray(folder/"absolute_difference.png",np.where(valid,diff,0))
    Image.fromarray(np.uint8(valid)*255).save(folder/"valid_support.png")
    fig,axes=plt.subplots(2,3,figsize=(17,8),layout="constrained")
    panels=[
        (original,"Query original: native center crop","gray",0,1),
        (aligned[i],"Aligned query: "+records[i]["selected_method"],"gray",0,1),
        (pseudo,"Median of OTHER aligned images (not clean GT)","gray",0,1),
        (np.where(valid,diff,np.nan),"Absolute difference + GT bboxes (display 0..64 DN)","magma",0,64/255),
        (np.where(valid,edge,np.nan),"Pseudo-Golden gradient magnitude (continuous)","gray",0,0.12),
        (np.where(valid,np.abs(original-pseudo),np.nan),"Unregistered query vs same Golden (display 0..64 DN)","magma",0,64/255),
    ]
    for ax,(data,title,cmap,lo,hi) in zip(axes.flat,panels):
        artist=ax.imshow(data,cmap=cmap,vmin=lo,vmax=hi)
        ax.set_title(title,fontsize=10)
        ax.set_xlabel("native common-canvas x [px]",fontsize=8)
        ax.set_ylabel("y [px]",fontsize=8)
        ax.tick_params(labelsize=7)
        if cmap=="magma":
            fig.colorbar(artist,ax=ax,shrink=0.7,label="absolute gray difference /255")
    colors={"p":"cyan","o":"lime","s":"red","sc":"deepskyblue"}
    for name,p in polygons:
        axes[1,0].add_patch(Polygon(p,fill=False,edgecolor=colors.get(name,"white"),linewidth=0.7))
    axes[1,0].set_xlim(0,original.shape[1])
    axes[1,0].set_ylim(original.shape[0],0)
    fig.suptitle(f"{gid} / {kind} / image {query['image_id']} / {query['original_filename']}\n"
                 f"LOO MAE {values['loo_mae_0_1']:.4f}; support {valid.mean():.1%}; "
                 "p=cyan, o=lime, s=red, sc=blue; no detection threshold",fontsize=12)
    fig.savefig(folder/"diagnostic.png",dpi=120)
    plt.close(fig)
    values["diagnostic_path"]=(folder/"diagnostic.png").relative_to(OUT).as_posix()
    return values


def register():
    summary=json.loads((OUT/"golden_feasibility_summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="GROUPED_PENDING_REGISTRATION"
    assert all(digest(ROOT/p)==h for p,h in summary["source_hashes"].items())
    rows={r["image_id"]:r for r in read_csv(OUT/"image_summary.csv")}
    annotations=defaultdict(list)
    for a in read_csv(AUDIT/"annotation_manifest.csv"):
        annotations[a["image_id"]].append(a)
    groups=read_csv(OUT/"group_summary.csv")
    pair_rows=[]
    reports=[]
    cv2.setRNGSeed(SEED)
    for group in groups:
        gid=group["group_id"]
        subset=[rows[i] for i in summary["registration_subset_image_ids"][gid]]
        h=min(int(r["height"]) for r in subset)
        w=min(int(r["width"]) for r in subset)
        crops=[((int(r["width"])-w)//2,(int(r["height"])-h)//2,w,h) for r in subset]
        raw=[load(r)[y:y+hh,x:x+ww].astype(np.float32)/255
             for r,(x,y,ww,hh) in zip(subset,crops)]
        ref=raw[0]
        identity=np.array([[1,0,0],[0,1,0]],dtype=np.float32)
        check,mask=warp(ref,identity)
        assert np.max(np.abs(check-ref))==0
        assert residual(ref,ref,mask)["mae_0_1"]==0
        aligned=[ref];masks=[np.ones_like(ref,dtype=bool)]
        forwards=[identity];kept=[subset[0]];kept_crops=[crops[0]]
        kept_records=[{"selected_method":"reference_identity"}]
        local_records=[]
        for i in range(1,len(raw)):
            mov=raw[i]
            ta,tm,tf,td=translation(ref,mov)
            sa,sm,sf,sd=similarity(ref,mov)
            record={
                "group_id":gid,"reference_image_id":subset[0]["image_id"],
                "query_image_id":subset[i]["image_id"],"filename":subset[i]["filename"],
                "original_filename":subset[i]["original_filename"],
                "native_common_width":w,"native_common_height":h,
                "reference_crop_x":crops[0][0],"reference_crop_y":crops[0][1],
                "query_crop_x":crops[i][0],"query_crop_y":crops[i][1],
                **td,**sd,"selected_method":"unavailable",
                "before_mae_0_1":"","before_edge_mae_0_1":"","before_normalized_edge_residual":"",
                "translation_mae_0_1":"","translation_edge_mae_0_1":"",
                "translation_normalized_edge_residual":"","similarity_mae_0_1":"",
                "similarity_edge_mae_0_1":"","similarity_normalized_edge_residual":"",
                "selected_mae_0_1":"","selected_normalized_edge_residual":"",
                "comparison_support_fraction":"","selected_forward_matrix":"",
                "note":"Optimizer convergence/model estimation is not proof of physical phase correspondence or Golden usability.",
            }
            common=tm.copy()
            if sm is not None:
                common &= sm
            if common.mean()<0.70:
                common=tm.copy()
                sa=sm=sf=None
                record["similarity_status"]="rejected_joint_support_below_70pct"
            if common.any():
                before=residual(ref,mov,common)
                after=residual(ref,ta,common)
                record.update({f"before_{k}":v for k,v in before.items()})
                record.update({f"translation_{k}":v for k,v in after.items()})
                selected=(ta,tm,tf,"translation")
                if sa is not None:
                    sr=residual(ref,sa,common)
                    record.update({f"similarity_{k}":v for k,v in sr.items()})
                    if sr["normalized_edge_residual"]<after["normalized_edge_residual"]:
                        selected=(sa,sm,sf,"bounded_similarity")
                selected_a,selected_m,selected_f,name=selected
                if selected_m.mean()>=0.70 and (td["translation_usable_support"] or name=="bounded_similarity"):
                    rr=residual(ref,selected_a,common)
                    record.update(selected_method=name,selected_mae_0_1=rr["mae_0_1"],
                        selected_normalized_edge_residual=rr["normalized_edge_residual"],
                        comparison_support_fraction=float(common.mean()),
                        selected_forward_matrix=json.dumps(selected_f.tolist()))
                    aligned.append(selected_a);masks.append(selected_m);forwards.append(selected_f)
                    kept.append(subset[i]);kept_crops.append(crops[i]);kept_records.append(record)
            local_records.append(record)
            pair_rows.append(record)
            print("REGISTRATION="+json.dumps({k:record[k] for k in (
                "group_id","query_image_id","translation_estimator_status","similarity_status",
                "selected_method","before_mae_0_1","selected_mae_0_1",
                "selected_normalized_edge_residual")}),flush=True)
        folder=OUT/"examples"/gid
        folder.mkdir()
        successful=[r for r in local_records if r["selected_method"]!="unavailable"]
        before=[r["before_mae_0_1"] for r in successful]
        after=[r["selected_mae_0_1"] for r in successful]
        edge=[r["selected_normalized_edge_residual"] for r in successful]
        group.update(
            tested_pairs=len(local_records),
            translation_ecc_converged=sum(r["translation_estimator_status"]=="ecc_converged" for r in local_records),
            translation_phase_only=sum(r["translation_estimator_status"]=="phase_only" for r in local_records),
            bounded_similarity_estimated=sum(r["similarity_status"]=="estimated_within_bounds" for r in local_records),
            selected_translation=sum(r["selected_method"]=="translation" for r in local_records),
            selected_similarity=sum(r["selected_method"]=="bounded_similarity" for r in local_records),
            registration_unavailable=sum(r["selected_method"]=="unavailable" for r in local_records),
            median_before_mae_0_1=float(np.median(before)) if before else "",
            median_after_mae_0_1=float(np.median(after)) if after else "",
            median_after_normalized_edge_residual=float(np.median(edge)) if edge else "",
            residual_reduced_pairs=sum(a<b for a,b in zip(after,before)),
            pseudo_golden_created=len(kept)>=3,
        )
        gr={"group_id":gid,"reference_image_id":subset[0]["image_id"],
            "selected_image_ids":[r["image_id"] for r in kept],
            "selected_native_crop_bounds":dict(zip((r["image_id"] for r in kept),kept_crops)),
            "pseudo_golden_is_clean_gt":False,"pseudo_golden_is_final_reference":False,
            "examples":[]}
        if len(kept)>=3:
            stack=np.stack([np.where(m,im,np.nan) for im,m in zip(aligned,masks)])
            pseudo=np.nanmedian(stack,axis=0)
            common=np.logical_and.reduce(masks)
            save_gray(folder/"pseudo_golden.png",np.where(common,pseudo,0))
            Image.fromarray(np.uint8(common)*255).save(folder/"pseudo_golden_support.png")
            gr["pseudo_golden_common_support_fraction"]=float(common.mean())
            worst=max(range(1,len(kept)),key=lambda i:kept_records[i]["selected_normalized_edge_residual"])
            cases=[("largest_tested_residual",worst)]
            negatives=[i for i in range(1,len(kept)) if int(kept[i]["total_annotations"])==0]
            if negatives and negatives[0]!=worst:
                cases.append(("zero_annotation_negative",negatives[0]))
            for kind,i in cases:
                gr["examples"].append(example(gid,i,kept,aligned,masks,forwards,kept_crops,
                                             annotations,folder,kept_records,kind))
        reports.append(gr)
        print("GROUP_COMPLETE="+gid,flush=True)
    write_csv(OUT/"registration_summary.csv",pair_rows)
    write_csv(OUT/"group_summary.csv",groups)
    assert len(pair_rows)==sum(int(g["tested_pairs"]) for g in groups)
    assert all(digest(ROOT/p)==h for p,h in summary["source_hashes"].items())
    summary.update(
        status="REGISTERED_PENDING_VISUAL_REVIEW",registration_groups=groups,
        group_examples=reports,registration_pair_count=len(pair_rows),
        registration_estimated_pairs=sum(r["selected_method"]!="unavailable" for r in pair_rows),
        translation_ecc_converged=sum(r["translation_estimator_status"]=="ecc_converged" for r in pair_rows),
        bounded_similarity_estimated=sum(r["similarity_status"]=="estimated_within_bounds" for r in pair_rows),
        pseudo_golden_groups=sum(bool(g["pseudo_golden_created"]) for g in groups),
        registration_protocol={
            "inputs":"Native grayscale, group-wide centered common crop, no pre-registration resize.",
            "translation":"One Hanning phase correlation initializer + one native translation ECC call, max 40 iterations, epsilon 1e-4; fixed sigma2 structural blur, no photometric correction. Phase-only result is explicitly distinguished when ECC fails.",
            "similarity":"One ORB(2000), Hamming ratio0.75, partial-affine RANSAC(3px,2000 iterations,0.99 confidence); only uniform scale0.95..1.05 and rotation+/-3deg. No shear, perspective or dense warp.",
            "support":"At least70% common valid native canvas for comparison; exclude interpolated borders. Preset registration feasibility guard, not a defect threshold.",
            "selection":"Compare translation vs bounded similarity on SAME support; lower normalized continuous Sobel residual is shown. No GT-guided transform selection.",
            "metrics":"Gray MAE in 0..1; continuous Sobel edge MAE; edge residual divided by mean reference gradient magnitude. No precision/recall/IoU or detection threshold.",
            "pseudo":"Median of aligned selected subset only, valid common support saved. Example query is excluded from its own leave-one-out median; at least two other contributors. Main median includes all kept subset images and is not evaluated as clean GT.",
            "sampling_warning":"Only up to6 representative native images per candidate group tested; results do not establish all629-image registration success.",
            "display":"Source/registration arrays not contrast corrected; difference plot common0..64 DN display range is visualization only, raw absolute difference PNG retained without threshold.",
        },
        source_files_unchanged=True,annotation_modifications=False,split_modifications=False,
        training_performed=False,detection_metrics_computed=False,
    )
    write_json(OUT/"golden_feasibility_summary.json",summary)
    print("REGISTRATION_DONE="+json.dumps({k:summary[k] for k in (
        "registration_pair_count","registration_estimated_pairs","translation_ecc_converged",
        "bounded_similarity_estimated","pseudo_golden_groups")}),flush=True)


def finalize():
    """Save the explicitly inspected sheet/diagnostic evidence, not a threshold verdict."""
    summary=json.loads((OUT/"golden_feasibility_summary.json").read_text(encoding="utf-8"))
    assert summary["status"]=="REGISTERED_PENDING_VISUAL_REVIEW"
    images=read_csv(OUT/"image_summary.csv")
    groups=read_csv(OUT/"group_summary.csv")
    pairs=read_csv(OUT/"registration_summary.csv")
    assert len(images)==629 and len({r["image_id"] for r in images})==629
    assert len({r["filename"] for r in images})==629
    assert sum(int(g["image_count"]) for g in groups)==629
    assert Counter(r["candidate_group"] for r in images)==Counter(summary["group_image_counts"])
    assert len(pairs)==15 and len({r["query_image_id"] for r in pairs})==15
    fixed={r["image_id"]:r for r in read_csv(SPLIT)}
    assert all(r["split"]==fixed[r["image_id"]]["split"] and
               r["capture_group"]==fixed[r["image_id"]]["group_key"] and
               r["filename"]==fixed[r["image_id"]]["filename"] for r in images)
    paths=[]
    for group in summary["group_examples"]:
        for e in group["examples"]:
            path=OUT/e["diagnostic_path"]
            assert path.is_file() and e["query_excluded_from_golden"]
            paths.append(e["diagnostic_path"])
    assert len(paths)==5
    for path in OUT.rglob("*.png"):
        with Image.open(path) as im:
            im.verify()
    assert all(digest(ROOT/p)==h for p,h in summary["source_hashes"].items())
    summary.update(
        status="COMPLETED",analysis_success=True,
        final_judgment="C",final_judgment_label="Not suitable",
        judgment_scope="Current 629-image exploratory grouping plus tested fixed translation / bounded-ORB-similarity protocol. Not proof that every specialized Golden method is impossible.",
        reasons=[
            "The three descriptor clusters remain coarse: U-turn location, framing, UI and visibility vary inside groups; same cluster does not imply a shareable clean reference.",
            "Translation reduces gray MAE but all three group medians retain normalized edge residual near one; extensive normal-pattern boundary differences remain visually.",
            "Two upstream human-confirmed zero-annotation negatives still show full-height line-edge differences against leave-one-out medians; some annotated defects are visible but not reliably isolated from structure artifacts.",
            "No bounded similarity estimate was obtained and no clean pseudo-Golden was certified. Median generation alone is not evidence of detection feasibility.",
        ],
        visual_review={
            "reviewer":"assistant_visual_inspection; feasibility evidence, not independent human annotation",
            "contact_sheets_inspected":["contact_sheets/G01.png","contact_sheets/G02.png","contact_sheets/G03.png"],
            "diagnostics_inspected":paths,
            "G01":"Mostly U-turn-at-top family but contact sheet also has straight-only, bottom-turn and severely cropped frames. Negative image372 still has repeated bright full-height boundary residuals. Image144 residual is broadly stronger at normal finger boundaries than many small GT-box contents.",
            "G02":"Many UI-bordered exports, but low-contrast/blank or different-turn frames also occur. Negative image7 shows bright full-height line residuals. Image543 differs in line visibility and global optical appearance; residual is not confined to its GT boxes.",
            "G03":"Higher line density is recognizable, but framing/turns and local line shape still vary. Image48 has visible particle signals inside GT boxes, alongside large repeated line-edge artifacts over the field.",
            "artifact_vs_defect":"Clear evidence of substantial non-defect structure residual in the two reviewed negatives. Some annotated objects have stronger local signal (e.g. image48); do not claim every defect is weaker than every artifact.",
            "scale_causality":"Residuals can arise from alignment, spacing/width, framing and photometric differences. This test does not isolate resize/scale as the unique cause or infer acquisition calibration.",
            "similarity_failures":"Rejected scale/rotation fits are failed correspondence estimates, often degenerate, NOT measured physical scales or real image rotations.",
        },
        registration_physical_correspondence_confirmed=False,
        detector_success_rate=None,
        pseudo_golden_certified_clean=False,
        errors=[],
        warnings=[
            "One native translation ECC call did not converge (G02 image543); only its phase-correlation translation is retained and clearly marked.",
            "Similarity: 13 partial-affine estimates rejected outside fixed small-transform bounds; 2 pairs lacked enough correspondences; 0 usable bounded similarity estimates.",
            "Coarse candidate groups are not canonical image types or guaranteed Golden-sharing groups; K3/K4 descriptor silhouette values are close.",
            "Only18 representative images /15 pairs were registered, not all629. All629 images were structurally exposed, including existing Val/Test.",
            "Medians are contaminated/uncertified feasibility references; query-excluded comparisons mitigate self-inclusion, not all reference bias.",
        ],
        qa={
            "image_summary_rows":629,"image_ids_unique":True,"image_files_present":True,
            "group_assignments_complete":True,"registration_pairs":15,
            "contact_sheets":3,"contact_sheet_cards":36,"registered_subset_images":18,
            "diagnostic_pngs":5,"png_integrity_verified":True,
            "protected_source_hash_count":len(summary["source_hashes"]),
            "all_protected_source_hashes_unchanged":True,
            "fixed_split_assignments_unchanged":True,
            "no_annotation_or_production_changes":True,
            "detection_thresholds_or_detection_metrics":False,
        },
    )
    write_json(OUT/"golden_feasibility_summary.json",summary)
    table=["| Group | Images | Native width / height range | Resolutions | Median period estimate (px) |",
           "|---|---:|---|---:|---:|"]
    registration=["| Group | Pairs | ECC converged | Phase only | Bounded similarity usable | Median gray MAE before / after | Median normalized edge residual after |",
                  "|---|---:|---:|---:|---:|---|---:|"]
    for g in groups:
        table.append(f"| {g['group_id']} | {g['image_count']} | {g['width_min']}..{g['width_max']} / {g['height_min']}..{g['height_max']} | {g['distinct_resolutions']} | {float(g['period_px_median']):.3f} |")
        registration.append(f"| {g['group_id']} | {g['tested_pairs']} | {g['translation_ecc_converged']} | {g['translation_phase_only']} | {g['bounded_similarity_estimated']} | {float(g['median_before_mae_0_1']):.5f} / {float(g['median_after_mae_0_1']):.5f} | {float(g['median_after_normalized_edge_residual']):.5f} |")
    examples=["| Group / query | Annotations | LOO gray MAE | MAE inside GT bbox | MAE outside GT bbox | Gradient-weighted difference outside bbox |",
              "|---|---:|---:|---:|---:|---:|"]
    for group in summary["group_examples"]:
        for e in group["examples"]:
            fmt=lambda v:"N/A" if v is None else f"{v:.5f}"
            examples.append(f"| {group['group_id']} / {e['query_image_id']} ({e['example_kind']}) | {e['query_annotation_count']} | {fmt(e['loo_mae_0_1'])} | {fmt(e['loo_bbox_inside_mae_0_1'])} | {fmt(e['loo_bbox_outside_mae_0_1'])} | {fmt(e['loo_bbox_outside_gradient_weighted_absdiff_0_1'])} |")
    content=f"""# Golden reference feasibility v1

## Scope and final judgment

**C - Not suitable for the current dataset-wide workflow with the tested grouping and simple registration protocol.**
This is a feasibility result, not a defect detector evaluation, and does not rule out every specialized Golden approach. No threshold tuning, detection Precision/Recall/IoU, learning, augmentation, annotation edits or split changes were performed.

All629 current images were described once. Registration was limited to18 actual images (6/group),15 nonidentity pairs. Existing Val/Test assignments were preserved but their images are now exploratory exposed data for this Golden review, not an untouched Golden evaluation set.

## Inputs and traceability

- Existing image/annotation metadata: data/derived/gt_629_audit_v1/image_manifest.csv and annotation_manifest.csv.
- Capture groups and fixed assignments: data/derived/fixed_split_v1/split_manifest.csv.
- Unchanged native image bytes: data/derived/yolov8_detection_fixed_v1/ images.
- Original filename, wafer, position, exposure and capture group use existing metadata; unknown values are not guessed.
- Annotation classes/counts are reporting and sample-diversity metadata only, never inputs to grouping or registration.

## Exploratory structural groups

One Ward hierarchy uses six equal-normalized descriptor blocks: native geometry, orientation tensor, periodicity, row intensity/gradient profiles, coarse spatial/spectral layout and gray statistics. Descriptor-only downsampling does not enter registration.
K was selected by maximum descriptor silhouette over2..20, not forced to7: K3=0.36057, K4=0.35855, K2=0.33583. The near tie with K4 means3 is an exploratory candidate count, not a definitive number of physical image types.

{chr(10).join(table)}

G01 has mostly unbordered exports and G02 many UI-bordered images; native dimension ranges alone do not establish acquisition scale or resize provenance. G03 has visibly denser line structure. All three contain multiple position labels and different U-turn/framing conditions.
The36 representative contact-sheet cards reveal blank/low-contrast, straight-only, top/bottom-turn and severe partial-pattern cases within coarse groups. Same cluster membership is not enough to certify one shared Golden.
Period estimates may select harmonics or near-DC structure in blank/cropped cases (including ~1150px extremes); they are descriptive features, not pitch metrology or calibration.

## Fixed registration protocol

Use group-wide minimum dimensions and record centered native crop offsets. No pre-registration resize or pixel-scale matching is applied.
Translation: one phase-correlation initialization with Hanning window, followed by native translation ECC (40 iterations, epsilon1e-4; fixed sigma2 structural blur).
Similarity: one ORB2000/Hamming ratio0.75/partial-affine RANSAC attempt; scale must lie0.95..1.05 and rotation within+/-3deg, with at least70% valid common support. No affine shear, perspective or dense warp.
These are fixed registration settings, not tuned defect thresholds. Bilinear subpixel warping is explicit and only affects derived outputs.
Both transforms, when available, are compared on the same intersection support; continuous Sobel residual selects the displayed diagnostic transform. GT is not used for registration or selection.

{chr(10).join(registration)}

Gray MAE is absolute grayscale difference divided by255. Normalized edge residual is mean absolute signed Sobel-x/y difference divided by mean reference gradient magnitude on the same valid support.
The15 gray MAEs decreased, but post-registration normalized edge residual remains approximately0.91..1.13 across pairs: gray-MAE reduction is not proof of aligned physical finger edges.
ECC convergence is an optimizer outcome, not a verified physical/periodic phase correspondence or detector success rate.
Similarity did not produce any usable bounded transform:13 fits violate bounds and2 lack enough correspondences. Several fits are degenerate. These values are failed feature correspondences, NOT evidence that the actual images rotated by large angles or had zero physical scale. No alternative matcher or parameter sweep was attempted.

## Pseudo-Golden candidates and artifact review

Each group has a median of6 aligned images, with common-support masks (~91.8%..96.5%). These medians are neither clean normal GT nor final production references.
Every displayed query is excluded from its own comparison median; at least two other contributors remain. Query-excluded comparisons avoid self-inclusion but do not remove contamination or structural-reference bias.
Native derived original/aligned/median/difference/support PNGs are stored per example. Difference PNG values are unthresholded absolute grayscale differences; diagnostic colormaps use a common0..64 DN display range only.

{chr(10).join(examples)}

Inside/outside-GT values are descriptive pixel averages, not detection metrics. GT bboxes include background, unannotated pixels are not automatically normal, and no defect mask or thresholded prediction was created.

Observed directly in the saved diagnostic images:

- G01 image372 (w5_22_lt.png), an upstream human-confirmed zero-label negative: full-height bright line-boundary residuals remain after registration; LOO MAE0.03045.
- G02 image7 (w1_24_c_png.jpg), another such negative: bright repeated line edges remain against its LOO median; LOO MAE0.03015.
- G01 image144: many GT particle boxes have weak interiors while normal finger-edge differences extend through the image.
- G02 image543: faint line visibility/optical appearance differs strongly from reference; difference is not restricted to annotated regions and ECC does not converge.
- G03 image48: some annotated particle signals are clearly visible, but extensive normal line-edge residual remains too. This does not support a claim that every defect signal is weaker than every artifact.

The negatives provide direct evidence that substantial difference can be caused without a true query defect. Edge placement, line width/spacing, framing and illumination/contrast can all contribute. This test does not identify resize or scale as the sole cause.

## Reasons for C

1. Coarse groups retain materially different framing, turns and optical appearance; a common reference is not established.
2. Translation reduces global difference without removing widespread normal-pattern boundary residuals.
3. Two reviewed true-negative controls still show extensive edge difference; local defects are not reliably isolated from these artifacts.
4. Median candidates can be generated technically, but no group has demonstrated an adequately clean, stable Golden under this fixed test.

This concludes the requested feasibility scope. No Golden detector, further grouping/registration optimization or follow-up experiment was implemented.

## Integrity and files

image_summary.csv:629 unique images, complete group assignment and existing metadata.
group_summary.csv:3 candidate groups, populations, resolutions and tested-registration summaries.
registration_summary.csv:15 native pair attempts, estimator states, crop offsets, transforms and residuals.
golden_feasibility_summary.json:method, image IDs, source hashes, metrics, manual visual notes, limitations and QA.
contact_sheets/:3 group sheets,12 cards each.
examples/:3 median candidates and5 query-excluded diagnostic examples, including2 negative controls.
All{len(summary['source_hashes'])} protected image/label/metadata hashes match their preparation values. Existing raw data, annotation, fixed splits, production code/config/dashboard and previous experiment outputs were not edited. No commit/push.

Warnings:1 ECC nonconvergence and15 unusable similarity attempts are preserved as diagnostic findings, not hidden or auto-corrected. No fatal runtime error.
"""
    (OUT/"golden_feasibility_report.md").write_text(content,encoding="utf-8")
    assert all(digest(ROOT/p)==h for p,h in summary["source_hashes"].items())
    print(json.dumps({k:summary[k] for k in (
        "status","total_images","candidate_group_count","group_image_counts","final_judgment",
        "registration_pair_count","translation_ecc_converged","bounded_similarity_estimated",
        "pseudo_golden_groups","qa","warnings")},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register",action="store_true")
    parser.add_argument("--finalize",action="store_true")
    args=parser.parse_args()
    assert not (args.register and args.finalize)
    finalize() if args.finalize else register() if args.register else prepare()
