; ModuleID = 'input.ll'
source_filename = "LLVMDialectModule"

; Function Attrs: nofree norecurse nosync nounwind memory(argmem: readwrite)
define void @main_kernel(ptr nocapture readonly %0, ptr nocapture readonly %1, ptr nocapture %2) local_unnamed_addr #0 {
  %4 = getelementptr i8, ptr %0, i64 4
  %5 = getelementptr i8, ptr %0, i64 8
  %6 = getelementptr i8, ptr %0, i64 12
  %7 = getelementptr i8, ptr %0, i64 16
  %8 = getelementptr i8, ptr %0, i64 20
  %9 = getelementptr i8, ptr %0, i64 24
  %10 = getelementptr i8, ptr %0, i64 28
  br label %.preheader2

.preheader2:                                      ; preds = %3, %57
  %11 = phi i64 [ 0, %3 ], [ %58, %57 ]
  %.idx1 = shl nuw nsw i64 %11, 4
  %12 = getelementptr i8, ptr %2, i64 %.idx1
  %.idx3 = shl nuw nsw i64 %11, 5
  %13 = getelementptr i8, ptr %0, i64 %.idx3
  %14 = getelementptr i8, ptr %4, i64 %.idx3
  %15 = getelementptr i8, ptr %5, i64 %.idx3
  %16 = getelementptr i8, ptr %6, i64 %.idx3
  %17 = getelementptr i8, ptr %7, i64 %.idx3
  %18 = getelementptr i8, ptr %8, i64 %.idx3
  %19 = getelementptr i8, ptr %9, i64 %.idx3
  %20 = getelementptr i8, ptr %10, i64 %.idx3
  br label %.preheader

.preheader:                                       ; preds = %.preheader2, %.preheader
  %21 = phi i64 [ 0, %.preheader2 ], [ %55, %.preheader ]
  %invariant.gep = getelementptr float, ptr %1, i64 %21
  %22 = getelementptr float, ptr %12, i64 %21
  %.promoted = load float, ptr %22, align 4
  %23 = load float, ptr %13, align 4
  %24 = load float, ptr %invariant.gep, align 4
  %25 = fmul float %23, %24
  %26 = fadd float %.promoted, %25
  store float %26, ptr %22, align 4
  %27 = load float, ptr %14, align 4
  %gep.1 = getelementptr i8, ptr %invariant.gep, i64 16
  %28 = load float, ptr %gep.1, align 4
  %29 = fmul float %27, %28
  %30 = fadd float %26, %29
  store float %30, ptr %22, align 4
  %31 = load float, ptr %15, align 4
  %gep.2 = getelementptr i8, ptr %invariant.gep, i64 32
  %32 = load float, ptr %gep.2, align 4
  %33 = fmul float %31, %32
  %34 = fadd float %30, %33
  store float %34, ptr %22, align 4
  %35 = load float, ptr %16, align 4
  %gep.3 = getelementptr i8, ptr %invariant.gep, i64 48
  %36 = load float, ptr %gep.3, align 4
  %37 = fmul float %35, %36
  %38 = fadd float %34, %37
  store float %38, ptr %22, align 4
  %39 = load float, ptr %17, align 4
  %gep.4 = getelementptr i8, ptr %invariant.gep, i64 64
  %40 = load float, ptr %gep.4, align 4
  %41 = fmul float %39, %40
  %42 = fadd float %38, %41
  store float %42, ptr %22, align 4
  %43 = load float, ptr %18, align 4
  %gep.5 = getelementptr i8, ptr %invariant.gep, i64 80
  %44 = load float, ptr %gep.5, align 4
  %45 = fmul float %43, %44
  %46 = fadd float %42, %45
  store float %46, ptr %22, align 4
  %47 = load float, ptr %19, align 4
  %gep.6 = getelementptr i8, ptr %invariant.gep, i64 96
  %48 = load float, ptr %gep.6, align 4
  %49 = fmul float %47, %48
  %50 = fadd float %46, %49
  store float %50, ptr %22, align 4
  %51 = load float, ptr %20, align 4
  %gep.7 = getelementptr i8, ptr %invariant.gep, i64 112
  %52 = load float, ptr %gep.7, align 4
  %53 = fmul float %51, %52
  %54 = fadd float %50, %53
  store float %54, ptr %22, align 4
  %55 = add nuw nsw i64 %21, 1
  %56 = icmp ult i64 %21, 3
  br i1 %56, label %.preheader, label %57

57:                                               ; preds = %.preheader
  %58 = add nuw nsw i64 %11, 1
  %59 = icmp ult i64 %11, 3
  br i1 %59, label %.preheader2, label %60

60:                                               ; preds = %57
  ret void
}

attributes #0 = { nofree norecurse nosync nounwind memory(argmem: readwrite) }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"Debug Info Version", i32 3}
